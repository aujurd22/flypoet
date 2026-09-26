"""Analyze two_ts arms vs baselines: CL readouts + signal statistics."""
import json, glob, os
import numpy as np

L = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_v2')

def show(tag, fn):
    p = os.path.join(L, fn)
    if not os.path.exists(p):
        print(f'{tag}: MISSING'); return
    d = json.load(open(p))
    print(f"{tag}: seed={d['seed']} forget={d['avg_forgetting']:+.4f} improve={d['avg_improvement']:+.4f}")
    for h in d['history']:
        if h['trained_on']:
            ev = ' '.join(f"{k}={v:.3f}" for k, v in sorted(h['evals'].items()))
            print(f"   after {h['trained_on']:<11}: {ev}")

print('=== baselines (existing runs) ===')
show('fly loss-gate', 'cl_fly_std_result.json')
show('skip random  ', 'cl_skip_std_result.json')
print('=== two_ts arms (new) ===')
for fn in sorted(glob.glob(os.path.join(L, 'cl_fly_std_gt-two_ts*_result.json'))):
    if '_s99' in fn: continue
    show(os.path.basename(fn).replace('cl_fly_std_gt-', '').replace('_result.json', ''),
         os.path.basename(fn))

print('\n=== signal stats from siglog (W250 arm) ===')
sig = os.path.join(L, 'cl_fly_std_gt-two_ts_siglog.jsonl')
if os.path.exists(sig):
    rows = [json.loads(l) for l in open(sig)]
    loss = np.array([r['loss'] for r in rows]); score = np.array([r['score'] for r in rows])
    allow = np.array([r['allow'] for r in rows])
    from scipy.stats import spearmanr
    ok = score > 0
    print(f'n={len(rows)} allow_rate={allow.mean():.3f} score_zero_frac={(score==0).mean():.3f}')
    if ok.sum() > 10:
        print(f'Spearman(loss, score | score>0) = {spearmanr(loss[ok], score[ok]).statistic:.3f}')
    for st in sorted(set(r['stage'] for r in rows)):
        a = np.array([r['allow'] for r in rows if r['stage'] == st])
        z = np.array([r['score'] == 0 for r in rows if r['stage'] == st])
        print(f'  stage {st}: allow={a.mean():.3f} zero-score={z.mean():.3f}')
