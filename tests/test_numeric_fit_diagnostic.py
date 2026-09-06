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
