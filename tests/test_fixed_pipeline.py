from copy import deepcopy
import json
import math
import pytest
from reactgdiff.eval.structured import pair_metrics, edits, anchors
from reactgdiff.pipeline.contracts import input_record, discrete_slots, requests, parameter_prompt, parse_proposal, fill_values, validate


def record():
    return {'index': 1, 'actions': 'ADD $1$ (987.25 g).', '_features': {'secret':987},
            'source':'answer 987.25 g', 'extracted_molecules':{'CCO':'$1$'},
            'extracted_temperature':{},'extracted_duration':{}}


def slots():
    return [{'operation_type':'ADD','material_refs':['$1$'], 'raw_text':'SECRET',
             'argument_text':'SECRET', 'quantity_slots':[{'value':987.25,'text':'SECRET','candidate_id':'NUM_987','slot_id':0,'unit':'g','numeric_type':'amount'}]}]


def test_target_and_numeric_predictions_cannot_reach_prompt():
    s=discrete_slots(slots()); text=parameter_prompt(record(),s,requests(s)[0])
    assert all(x not in text for x in ('987','SECRET','candidate_id','actions','_features','answer'))
    changed=record();changed['actions']='FILTER';changed['source']='different'
    assert parameter_prompt(changed,s,requests(s)[0]) == text


def test_source_policy_is_explicit():
    assert 'source' not in input_record(record())
    assert input_record(record(),True)['source']==record()['source']


@pytest.mark.parametrize('text', ['NaN','inf','1e999','3 g','answer: 3','ABSTAIN',''])
def test_invalid_numeric_generation_abstains(text):
    assert parse_proposal(text) is None


def test_numeric_generation_not_restricted_to_candidate_pointer():
    assert parse_proposal('1.25e-2') == .0125
    result=fill_values(discrete_slots(slots()),[.0125])
    assert result[0]['quantity_slots'][0]['value']==.0125
    assert validate(record(),result)['unsupported_parameter_count']==1


def test_missing_values_remain_visible_and_rejected():
    result=fill_values(slots(),[None])
    assert result[0]['quantity_slots'][0]['text']=='<NUMERIC_SLOT_MISSING>'
    assert validate(record(),result)['status']=='ABSTAIN'


def test_gate_catches_invalid_binding_and_physics():
    result=fill_values(slots(),[-3]);result[0]['material_refs']=['$99$']
    rules={x['rule'] for x in validate(record(),result)['issues']}
    assert rules=={'negative_non_temperature','unresolved_material'}


def test_sourced_rule_applies_only_to_matching_operation_and_unit():
    result=fill_values(slots(),[3])
    rule={'id':'test-only','source':'synthetic test fixture','operation':'ADD','unit':'g','max':2}
    assert validate(record(),result,rules=[rule])['status']=='ABSTAIN'
    rule['operation']='FILTER'
    assert validate(record(),result,rules=[rule])['status']=='PASS'
    with pytest.raises(ValueError): validate(record(),result,rules=[{'max':2}])


def test_matching_step_count_does_not_hide_reordering():
    m=pair_metrics('FILTER ; ADD $1$.','ADD $1$ ; FILTER.')
    assert m['thoth_step_m']==1 and m['thoth_order_s']==0
    assert m['thoth_order_lcs']==.5 and m['occurrence_order_tau']==-1


def test_repeated_operation_parameter_swap_is_detected():
    p='ADD $1$ (2 g) ; ADD $1$ (1 g).'
    r='ADD $1$ (1 g) ; ADD $1$ (2 g).'
    m=pair_metrics(p,r)
    assert m['thoth_order_s']==1 and m['aligned_parameter_iou']==0


def test_wrong_object_cannot_receive_numeric_credit():
    m=pair_metrics('ADD $2$ (1 g).','ADD $1$ (1 g).')
    assert m['aligned_parameter_iou']==0


def test_empty_output_and_one_step_tau_conventions():
    assert pair_metrics('','')['thoth_step_m']==0
    assert pair_metrics('ADD.','ADD.')['thoth_order_tau_monotone']==0
    assert pair_metrics('ADD.','ADD.')['thoth_semantic_a_adapted']==2.5


def test_skeleton_error_directions():
    assert edits(['ADD'],['ADD','FILTER'])[0]['missing']==1
    assert edits(['ADD','FILTER'],['ADD'])[0]['extra']==1
    assert edits(['ADD'],['FILTER'])[0]['substitution']==1


def test_fill_does_not_mutate_or_change_operations():
    original=slots();saved=deepcopy(original)
    filled=fill_values(original,[4])
    assert original==saved
    assert [s['operation_type'] for s in filled]==['ADD']
    with pytest.raises(ValueError): fill_values(original,[])


def test_malformed_external_value_is_rejected_without_crashing():
    result=fill_values(slots(),[1])
    result[0]["quantity_slots"][0]["value"]="bad"
    assert validate(record(),result,True)["status"]=="ABSTAIN"
