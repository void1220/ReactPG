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
    config.save_pretrained(base_path)
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
          '--device','cpu','--limit','1','--parameter-train-records','1','--batch-size','1',
          '--sample-steps','2','--quantity-threshold','0','--output',str(output)])
    main()
    result=json.loads((output/'report.json').read_text())
    assert result['allow_operation_completion'] is False
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
    from reactgdiff.pipeline.specialist import predict_skeleton
    from reactgdiff.pipeline.contracts import input_record
    fresh = predict_skeleton(sk, [input_record({**record, 'index': 2})], 'cpu', 1)
    assert len(fresh) == 1 and fresh[0]['index'] == 2
    assert 'predicted_skeleton' in fresh[0]
