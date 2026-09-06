"""Fixed-structure continuous-slot contracts and reference-free validity gate."""
from copy import deepcopy
import json
import math
import re
from reactgdiff.data.numeric_evidence import normalize_unit, numeric_candidates_from_record, infer_numeric_type, parse_numeric_value_unit
from reactgdiff.data.action_parser import KNOWN_OPENEXP_ACTIONS

INPUT_FIELDS = ('index', 'REACTANT', 'PRODUCT', 'CATALYST', 'SOLVENT', 'extracted_molecules', 'molecules', 'extracted_duration', 'extracted_temperature')


def input_record(record, include_source=False):
    result = {key: deepcopy(record[key]) for key in INPUT_FIELDS if key in record}
    if include_source: result['source'] = str(record.get('source', ''))
    return result


def discrete_slots(slots):
    """Whitelist prevents target argument text, numeric pointers and values leaking."""
    out = []
    for i, slot in enumerate(slots):
        refs = list(slot.get('material_refs') or [])
        out.append({'step_id': i, 'operation_type': slot['operation_type'],
                    'material_refs': refs, 'material_ref': refs[0] if refs else '<NONE>',
                    'condition': slot.get('condition', '<NONE>'),
                    'duration_ref': slot.get('duration_ref', ''),
                    'temperature_ref': slot.get('temperature_ref', ''),
                    'condition_values': {},
                    'quantity_slots': [dict(slot_id=q.get('slot_id', j), unit=q['unit'],
                                            numeric_type=q.get('numeric_type', 'amount'), value=None,
                                            text='<NUMERIC_SLOT_MISSING>', source='unfilled')
                                       for j, q in enumerate(slot.get('quantity_slots') or [])]})
    return out


def requests(slots):
    return [dict(id=f'{i}:{j}', step=i, quantity=j, unit=q['unit'], operation=s['operation_type'])
            for i, s in enumerate(slots) for j, q in enumerate(s['quantity_slots'])]


def parameter_prompt(record, slots, request, include_source=False):
    # Put requested slot first so a truncated graph cannot remove its identity.
    payload = {'request': request, 'input': input_record(record, include_source), 'discrete_steps': discrete_slots(slots)}
    return 'Predict only the numeric value in the requested unit, or ABSTAIN.\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def parse_proposal(text):
    text = text.strip()
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', text): return None
    value = float(text)
    return value if math.isfinite(value) else None


def fill_values(slots, values):
    out = discrete_slots(slots)
    reqs = requests(out)
    if len(reqs) != len(values): raise ValueError('Parameter output count mismatch')
    for request, value in zip(reqs, values):
        q = out[request['step']]['quantity_slots'][request['quantity']]
        q.update(value=value, text=f'{value:.8g} {q["unit"]}' if value is not None else '<NUMERIC_SLOT_MISSING>', source='specialist_seq2seq')
    return out


def validate(record, slots, include_source=False, rules=()):
    """Physical/type checks only. PASS is not a wet-lab safety certificate."""
    issues, unsupported, total = [], 0, 0
    applied_rules = []
    for rule in rules:
        if not all(rule.get(k) for k in ('id', 'source', 'operation', 'unit')):
            raise ValueError('Each supplied rule requires id, source, operation, and unit')
        if not any(k in rule for k in ('min', 'max')):
            raise ValueError('Rule has no bounds')
        if any(not isinstance(rule[k], (int,float)) or not math.isfinite(rule[k]) for k in ('min','max') if k in rule):
            raise ValueError('Invalid rule bound')
        if rule.get('min', -math.inf) > rule.get('max', math.inf): raise ValueError('Reversed rule bounds')
    materials = set((record.get('extracted_molecules') or {}).values())
    evidence = numeric_candidates_from_record(input_record(record, include_source), include_source=include_source)
    allowed = {'ug','mg','g','kg','ul','ml','l','umol','mmol','mol','eq','%','molar','normal','degc','h','min','s','day','times'}
    for i, slot in enumerate(slots):
        if slot.get('operation_type') not in KNOWN_OPENEXP_ACTIONS:
            issues.append({'step': i, 'rule': 'unknown_operation'})
        for ref in slot.get('material_refs', []):
            if ref not in materials: issues.append({'step': i, 'rule': 'unresolved_material', 'ref': ref})
        for kind, field in [('duration', 'extracted_duration'), ('temperature', 'extracted_temperature')]:
            ref = slot.get(kind+'_ref')
            if ref and ref not in set((record.get(field) or {}).values()):
                issues.append({'step': i, 'rule': 'unresolved_'+kind, 'ref': ref})
            if ref:
                for raw, candidate_ref in (record.get(field) or {}).items():
                    if candidate_ref != ref: continue
                    parsed = parse_numeric_value_unit(str(raw))
                    if parsed and ((kind == 'duration' and parsed[0] < 0) or (kind == 'temperature' and parsed[1] == 'degc' and parsed[0] < -273.15)):
                        issues.append({'step': i, 'rule': 'invalid_'+kind+'_evidence', 'ref': ref})
        for j, q in enumerate(slot.get('quantity_slots') or []):
            total += 1
            value, unit = q.get('value'), normalize_unit(q.get('unit', ''))
            issue = None
            numeric_valid = isinstance(value, (int,float)) and not isinstance(value, bool) and math.isfinite(value)
            if not numeric_valid: issue = 'missing_or_nonfinite'
            elif unit not in allowed: issue = 'unknown_unit'
            elif unit != 'degc' and value < 0: issue = 'negative_non_temperature'
            elif unit == 'degc' and value < -273.15: issue = 'below_absolute_zero'
            if not issue and q.get('numeric_type') not in (None, infer_numeric_type('', unit), 'yield' if unit == '%' else infer_numeric_type('', unit)):
                issue = 'type_unit_mismatch'
            for rule in rules:
                if rule['operation'] == slot['operation_type'] and normalize_unit(rule['unit']) == unit and numeric_valid:
                    applied_rules.append(rule['id'])
                    if not rule.get('min', -math.inf) <= value <= rule.get('max', math.inf):
                        issues.append({'step':i, 'slot':j, 'rule':'supplied_constraint', 'rule_id':rule['id']})
            if issue: issues.append({'step': i, 'slot': j, 'rule': issue})
            supported = numeric_valid and any(c.value is not None and normalize_unit(c.unit or '') == unit and math.isclose(c.value, value, rel_tol=1e-6, abs_tol=1e-8) for c in evidence)
            if not supported: unsupported += 1
    if not slots: issues.append({'rule': 'empty_procedure'})
    return {'status': 'ABSTAIN' if issues else 'PASS', 'issues': issues,
            'parameter_count': total, 'unsupported_parameter_count': unsupported,
            'scope': 'format_reference_and_physical_validity_only',
            'chemical_safety_verified': False, 'applied_rule_ids': applied_rules}
