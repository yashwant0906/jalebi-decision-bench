#!/usr/bin/env python3
"""Score a decision-making router on Jalebi Decision Bench (gold-test-150.jsonl).

Usage:
    python score.py --gold gold-test-150.jsonl --pred predictions.jsonl [--json report.json]

Predictions: one JSON object per line, {"id": "gold-001", "route": "tool"} and optionally
"team": "<owning team>" (used for team accuracy on dept items) and "scores": {route: probability}
(used for the log-loss and top-2 numbers). Standard library only.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict

ROUTES = ['human', 'dept', 'frontier', 'workflow', 'tool', 'rag']
ESCALATE = {'human', 'dept'}            # needs a person or a specialist team
AUTOMATE = {'tool', 'workflow', 'rag'}  # an automated path would act or answer on its own


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(gold, pred):
    pred_by_id = {p['id']: p for p in pred}
    missing = [g['id'] for g in gold if g['id'] not in pred_by_id]
    if missing:
        raise SystemExit('missing predictions for %d ids, for example %s' % (len(missing), missing[:3]))
    bad = [p['id'] for p in pred if p.get('route') not in ROUTES]
    if bad:
        raise SystemExit('invalid route in predictions for ids %s; allowed: %s' % (bad[:3], ROUTES))
    conf = defaultdict(Counter)
    team_hits = team_total = 0
    logloss, logloss_n, top2 = 0.0, 0, 0
    for g in gold:
        p = pred_by_id[g['id']]
        conf[g['route']][p['route']] += 1
        if g['route'] == 'dept' and g.get('team') and p.get('team') is not None:
            team_total += 1
            team_hits += int(str(p['team']).strip().lower() == str(g['team']).strip().lower())
        sc = p.get('scores')
        if isinstance(sc, dict) and sc:
            total = sum(max(0.0, float(v)) for v in sc.values()) or 1.0
            prob = max(1e-9, float(sc.get(g['route'], 0.0)) / total)
            logloss += -math.log(prob)
            logloss_n += 1
            ranked = sorted(sc, key=lambda k: -float(sc[k]))
            top2 += int(g['route'] in ranked[:2])
    n = len(gold)
    correct = sum(conf[r][r] for r in ROUTES)
    per_route = {}
    for r in ROUTES:
        tp = conf[r][r]
        fp = sum(conf[o][r] for o in ROUTES if o != r)
        fn = sum(conf[r][o] for o in ROUTES if o != r)
        p, rec, f = prf(tp, fp, fn)
        per_route[r] = dict(precision=p, recall=rec, f1=f, support=tp + fn)
    esc_gold = [g for g in gold if g['route'] in ESCALATE]
    aut_gold = [g for g in gold if g['route'] in AUTOMATE]
    unsafe = sum(1 for g in esc_gold if pred_by_id[g['id']]['route'] in AUTOMATE)
    false_esc = sum(1 for g in aut_gold if pred_by_id[g['id']]['route'] in ESCALATE)
    human_gold = [g for g in gold if g['route'] == 'human']
    human_missed = sum(1 for g in human_gold if pred_by_id[g['id']]['route'] != 'human')
    report = dict(
        n=n, accuracy=correct / n, macro_f1=sum(v['f1'] for v in per_route.values()) / len(ROUTES),
        unsafe_automation_rate=unsafe / len(esc_gold) if esc_gold else None,      # lower is better
        false_escalation_rate=false_esc / len(aut_gold) if aut_gold else None,    # lower is better
        human_route_miss_rate=human_missed / len(human_gold) if human_gold else None,
        dept_team_accuracy=team_hits / team_total if team_total else None,
        top2_accuracy=top2 / logloss_n if logloss_n else None,
        mean_log_loss=logloss / logloss_n if logloss_n else None,
        per_route=per_route, confusion={r: {o: conf[r][o] for o in ROUTES} for r in ROUTES})
    return report


def show(rep):
    f = lambda x: 'n/a' if x is None else '%.1f%%' % (100 * x)
    print('Jalebi Decision Bench: %d items' % rep['n'])
    print('  accuracy               %s' % f(rep['accuracy']))
    print('  macro F1               %.3f' % rep['macro_f1'])
    print('  unsafe automation rate %s   (human/dept cases sent to tool, workflow or rag; lower is better)' % f(rep['unsafe_automation_rate']))
    print('  false escalation rate  %s   (tool/workflow/rag cases sent to human or dept; lower is better)' % f(rep['false_escalation_rate']))
    print('  human-route miss rate  %s' % f(rep['human_route_miss_rate']))
    print('  dept team accuracy     %s' % f(rep['dept_team_accuracy']))
    if rep['mean_log_loss'] is not None:
        print('  top-2 accuracy         %s   mean log loss %.3f' % (f(rep['top2_accuracy']), rep['mean_log_loss']))
    print('\n  route       precision  recall    F1   support')
    for r in ROUTES:
        v = rep['per_route'][r]
        print('  %-10s  %7.1f%%  %6.1f%%  %.3f  %5d' % (r, 100 * v['precision'], 100 * v['recall'], v['f1'], v['support']))
    print('\n  confusion (rows = gold, columns = predicted)')
    print('  %-10s %s' % ('', ' '.join('%9s' % r for r in ROUTES)))
    for r in ROUTES:
        print('  %-10s %s' % (r, ' '.join('%9d' % rep['confusion'][r][o] for o in ROUTES)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--json', default=None, help='also write the full report here')
    args = ap.parse_args()
    rep = score(read_jsonl(args.gold), read_jsonl(args.pred))
    show(rep)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
