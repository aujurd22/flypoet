"""Final 3-seed aggregation for the two_ts vs loss-gate comparison."""
import json, glob, os
import numpy as np

L = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_v2')
NOISE = 0.05

def load(fn):
    p = os.path.join(L, fn)
    return json.load(open(p)) if os.path.exists(p) else None

arms = {
    'loss-gate': ['cl_fly_std_result.json', 'cl_fly_std_s8_result.json', 'cl_fly_std_s9_result.json'],
    'two_ts W250': ['cl_fly_std_gt-two_ts_result.json', 'cl_fly_std_gt-two_ts_s8_result.json',
                    'cl_fly_std_gt-two_ts_s9_result.json'],
    'two_tsc': ['cl_fly_std_gt-two_tsc_result.json'],
    'skip 70%': ['cl_skip_std_result.json'],
}
print(f"{'arm':<14} {'seeds':>5}  {'improve (mean±std)':<20} {'forget (mean±std)':<20} per-seed")
store = {}
for a, fns in arms.items():
    imps, fgs, detail = [], [], []
    for fn in fns:
        d = load(fn)
        if d is None:
            detail.append(f'{fn.split("_result")[0][-6:]}:missing'); continue
        imps.append(d['avg_improvement']); fgs.append(d['avg_forgetting'])
        detail.append(f"s{d['seed']}:{d['avg_improvement']:+.3f}")
    if not imps:
        print(f'{a:<14} none'); continue
    store[a] = (np.mean(imps), np.std(imps))
    print(f'{a:<14} {len(imps):>5}  {np.mean(imps):+.3f} ± {np.std(imps):.3f}       '
          f'{np.mean(fgs):+.3f} ± {np.std(fgs):.3f}       {" ".join(detail)}')

print('\nverdicts (|delta| > 0.05 = real):')
if 'two_ts W250' in store and 'loss-gate' in store:
    d = store['two_ts W250'][0] - store['loss-gate'][0]
    v = 'REAL' if abs(d) > NOISE else 'WITHIN NOISE'
    print(f'  two_ts W250 - loss-gate   = {d:+.3f}  -> {v}')
if 'two_tsc' in store and 'skip 70%' in store:
    d = store['two_tsc'][0] - store['skip 70%'][0]
    v = 'REAL' if abs(d) > NOISE else 'WITHIN NOISE'
    print(f'  two_tsc     - skip 70%   = {d:+.3f}  -> {v}')
if 'two_ts W250' in store and 'two_tsc' in store:
    d = store['two_tsc'][0] - store['two_ts W250'][0]
    v = 'REAL' if abs(d) > NOISE else 'WITHIN NOISE'
    print(f'  two_tsc     - two_ts raw = {d:+.3f}  -> {v}')
