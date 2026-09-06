"""Offline CPU integration: real tiny T5 and diffusion, no downloaded weights."""
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.joint_diffusion import ReactGDiffFeaturizer
from reactgdiff.models.procedure_graph_diffusion import ProcedureGraphDiffusion, save_procedure_graph_diffusion_checkpoint
from scripts.train_skeleton_seq2seq import SkeletonSeq2SeqModel
from reactgdiff.utils.io import write_jsonl


def test_full_fixed_pipeline_offline(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    from scripts.run_fixed_pipeline import main
    base_path = tmp_path/'base'
    vocab={'<pad>':0,'</s>':1,'<unk>':2,'5':3,'ADD':4,'YIELD':5,'g':6,'ABSTAIN':7}
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'))
    backend.pre_tokenizer=Whitespace()
    backend.post_processor=TemplateProcessing(single='$A </s>',special_tokens=[('</s>',1)])
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='</s>',unk_token='<unk>',model_input_names=['input_ids','attention_mask'])
    tokenizer.save_pretrained(base_path)
    config=T5Config(vocab_size=len(vocab),d_model=16,d_ff=32,d_kv=8,num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
    original = T5ForConditionalGeneration(config)
    original.save_pretrained(base_path)
    # Verify actual saved pretrained tensors are loaded; skeleton loading is forbidden here.
    from reactgdiff.pipeline import specialist
    with monkeypatch.context() as isolated:
        def forbidden(*args, **kwargs):
            raise AssertionError('Numeric initialization must not load skeleton weights')
        isolated.setattr(specialist, 'load_skeleton', forbidden)
        loaded, _ = specialist.load_parameter_base(str(base_path))
        assert all(torch.equal(value, loaded.state_dict()[key])
                   for key, value in original.state_dict().items())
        del loaded
    record={'index':1,'REACTANT':['CCO'],'PRODUCT':['CC=O'],'CATALYST':[],'SOLVENT':[],
            'extracted_molecules':{'CCO':'$1$','CC=O':'$-1$'},'extracted_duration':{},'extracted_temperature':{},
            'actions':'ADD $1$ (5 g) ; YIELD $-1$.','source':'5 g added'}
    codec=GraphTargetCodec.fit([record],max_steps=4,max_material_refs=2,max_material_slots=1)
    wrapper=SkeletonSeq2SeqModel(T5ForConditionalGeneration(config),hidden_size=16,max_steps=4,length_loss_weight=0)
    sk=tmp_path/'skeleton.pt'
    torch.save({'checkpoint_type':'seq2seq_skeleton','model_name':str(base_path),
                'model_state':wrapper.state_dict(),'codec':codec.to_dict(),'tokenizer_length':len(tokenizer),
                'action_tokens':{},'config':{'max_steps':4,'target_format':'natural_text','max_input_length':128,'max_target_length':8}},sk)
    feature=ReactGDiffFeaturizer.fit([record],field_dim=8)
    graph=ProcedureGraphDiffusion(condition_dim=feature.condition_dim,action_dim=codec.action_dim,
          material_dim=codec.material_dim,condition_slot_dim=codec.condition_dim,unit_dim=codec.unit_dim,
          max_steps=4,max_material_slots=1,hidden_dim=16,dit_depth=1,dit_heads=2,diffusion_steps=2,skeleton_conditioning=True)
    gp=tmp_path/'graph.pt'
    save_procedure_graph_diffusion_checkpoint(gp,model=graph,codec=codec,condition_featurizer=feature.to_dict(),history=[])
    train=tmp_path/'train.jsonl';val=tmp_path/'val.jsonl';cache=tmp_path/'cache.jsonl'
    write_jsonl(train,[{**record,'_split':'train'}])
    write_jsonl(val,[{**record,'index':2,'_split':'val'}])
    write_jsonl(cache,[{'index':2,'predicted_skeleton':['ADD','YIELD']}])
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['run_fixed_pipeline','--input',str(val),'--train',str(train),
          '--graph-checkpoint',str(gp),'--skeleton-checkpoint',str(sk),'--skeleton-cache',str(cache),
          '--parameter-base-model',str(base_path),'--device','cpu','--limit','1','--parameter-train-records','1','--batch-size','1',
          '--sample-steps','2','--quantity-threshold','0','--output',str(output)])
    main()
    result=json.loads((output/'report.json').read_text())
    assert result['allow_operation_completion'] is False
    assert result['parameter_training']['initialization_kind'] == 'original_pretrained'
    assert result['parameter_training']['initialization'] == str(base_path)
    assert result['parameter_training']['examples'] == 1
    assert result['parameter_training']['train_ids'] == [1]
    assert len(result['stages']) == 8
    assert all(x['count']==1 for x in result['stages'].values())
    assert result['stages']['compiler_surface_ORACLE']['exact_match_rate']==1
    assert result['data']['id_overlap']==0
    # Reloading the saved numerical model exercises the no-training path.
    output2=tmp_path/'run_reload'
    argv=list(sys.argv)
    argv[argv.index('--output')+1]=str(output2)
    argv += ['--parameter-model',str(output/'parameter_model')]
    monkeypatch.setattr(sys,'argv',argv)
    main()
    assert (output2/'report.json').is_file()
    # A previous skeleton-initialized filler must not silently enter this comparison.
    import pytest
    metadata_path = output/'parameter_model'/'parameter_training.json'
    old_meta = json.loads(metadata_path.read_text())
    old_meta.pop('initialization_kind')
    metadata_path.write_text(json.dumps(old_meta))
    argv[argv.index('--output')+1] = str(tmp_path/'reject_old')
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='original pretrained weights'):
        main()
    from reactgdiff.pipeline.specialist import predict_skeleton
    from reactgdiff.pipeline.contracts import input_record
    fresh = predict_skeleton(sk, [input_record({**record, 'index': 2})], 'cpu', 1)
    assert len(fresh) == 1 and fresh[0]['index'] == 2
    assert 'predicted_skeleton' in fresh[0]

    from scripts.run_numeric_fit_diagnostic import main as fit_main
    fit_output = tmp_path/'numeric_fit'
    monkeypatch.setattr(sys, 'argv', ['numeric_fit', '--train',str(train),'--validation',str(val),
        '--graph-checkpoint',str(gp),'--base-model',str(base_path),'--train-records','1',
        '--validation-records','1','--epochs','1','--batch-size','1','--device','cpu','--output',str(fit_output)])
    fit_main()
    fit_report = json.loads((fit_output/'report.json').read_text())
    assert len(fit_report['arms']) == 2
    for arm in fit_report['arms'].values():
        assert [x['epoch'] for x in arm['curve']] == [0,1]
        assert arm['training']['optimizer_steps'] == 1
        assert arm['curve'][1]['validation']['count'] == 1
    assert (fit_output/'summary.md').is_file()

    resume_output = tmp_path/'numeric_fit_resume'
    monkeypatch.setattr(sys, 'argv', ['numeric_fit', '--resume-run',str(fit_output),
        '--arm','assisted','--epochs','1','--output',str(resume_output)])
    fit_main()
    resumed = json.loads((resume_output/'report.json').read_text())
    assert list(resumed['arms']) == ['source_assisted_DIAGNOSTIC']
    arm = resumed['arms']['source_assisted_DIAGNOSTIC']
    assert [x['epoch'] for x in arm['curve']] == [1,2]
    assert arm['training']['initialization_kind'] == 'continued_diagnostic_weights'
    assert resumed['train_ids'] == fit_report['train_ids']
