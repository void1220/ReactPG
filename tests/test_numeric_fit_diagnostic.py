from scripts.run_numeric_fit_diagnostic import examples_from_records, summarize_predictions
from reactgdiff.models.graph_codec import GraphTargetCodec


def test_diagnostic_source_boundary():
    record = {'index':1,'REACTANT':['CCO'],'PRODUCT':['CC=O'],
              'extracted_molecules':{'CCO':'$1$','CC=O':'$-1$'},
              'actions':'ADD $1$ (7.125 g) ; YIELD $-1$.', 'source':'Added 7.125 g of material.'}
    codec = GraphTargetCodec.fit([record], max_steps=4, max_material_refs=2, max_material_slots=1)
    free = examples_from_records([record],codec,False)
    assisted = examples_from_records([record],codec,True)
    assert len(free) == len(assisted) == 1
    assert free[0]['target'] == assisted[0]['target'] == '7.125'
    assert '7.125' not in free[0]['prompt']
    assert '7.125' in assisted[0]['prompt']
    assert free[0]['matching_value_unit_in_input'] is False
    assert assisted[0]['matching_value_unit_in_input'] is True


def test_numeric_diagnostic_metrics_count_invalid_as_wrong():
    result = summarize_predictions([
        {'target':'10','raw':'10','value':10},
        {'target':'10','raw':'10.5','value':10.5},
        {'target':'10','raw':'nonsense','value':None}])
    assert result['valid_numeric_rate'] == 2/3
    assert result['numeric_exact_rate'] == 1/3
    assert result['within_10_percent_rate'] == 2/3


def test_material_binding_uses_structure_not_quantity_position():
    from reactgdiff.data.action_parser import quantity_material_bindings
    assert quantity_material_bindings('$1$ (5 g, 10 mmol), $2$ (7 g)') == ['$1$','$1$','$2$']
    assert quantity_material_bindings('$1$ and $2$ together (7 g)') == ['']
    assert quantity_material_bindings('$1$ (5 g), then (7 g)') == ['$1$','']


def test_bound_prompt_keeps_material_but_removes_target_value():
    import json
    from reactgdiff.pipeline.contracts import discrete_slots, requests, parameter_prompt
    record = {'index':1,'extracted_molecules':{'CCO':'$1$','O':'$2$'},
              'actions':'MAKESOLUTION $1$ (7.125 g, 9.875 mmol), $2$ (8.625 g).'}
    codec = GraphTargetCodec.fit([record],max_steps=4,max_material_refs=3,max_material_slots=4)
    slots = discrete_slots(codec.target_slots_from_record(record))
    prompt = parameter_prompt(record,slots,requests(slots)[2])
    context = json.loads(prompt.split('\n',1)[1])['requested_context']
    assert context['material_ref'] == '$2$'
    assert context['material_identifiers'] == ['O']
    assert context['binding'] == 'explicit_graph_binding'
    assert all(v not in prompt for v in ['7.125','9.875','8.625'])
    # Old predicted graphs do not possess these bindings; never assign by ordinal position.
    for q in slots[0]['quantity_slots']:
        q.pop('material_ref',None)
    context = json.loads(parameter_prompt(record,slots,requests(slots)[2]).split('\n',1)[1])['requested_context']
    assert context['material_ref'] is None and context['binding'] == 'unresolved'
