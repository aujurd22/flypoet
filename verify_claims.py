"""Credential script: recompute every headline claim from committed artifacts.

Run:  python verify_claims.py
Prints a PASS/FAIL table against pre-registered bounds and writes
logs_v2/credentials_report.json. If any line fails, the corresponding claim
in PAPER.md / REPORT_*.md must be corrected — claims and checks live or die
together.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(ROOT, "logs_v2")


def load(fp):
    return json.load(open(os.path.join(L, fp)))


def final_val(tag):
    return json.load(open(os.path.join(L, f"{tag}_final.json"))), \
        json.loads(open(os.path.join(L, f"{tag}_probes.jsonl"),
                        encoding="utf-8").read().strip().splitlines()[-1])


def main():
    checks = []

    def check(cid, ok, detail):
        checks.append({"id": cid, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid}: {detail}", flush=True)

    # ---- C1: 12k 3-seed (92.6M): k25 beats dense on both committed seeds
    for s in (8, 9):
        _, vd = final_val(f"std_s{s}")
        _, vk = final_val(f"flynetS_k25_s{s}")
        check(f"C1-92.6M-12k-seed{s}-k25-wins", vk["val"] < vd["val"],
              f"dense {vd['val']} vs k25 {vk['val']}")

    # ---- C2: scale ladder b8 points (216M / 334M): k25 beats dense
    _, vd = final_val("std_Lb8")
    _, vk = final_val("flynetS_Lb8k25")
    check("C2-216M-b8-k25-wins", vk["val"] < vd["val"],
          f"dense {vd['val']} vs k25 {vk['val']}")
    _, vd = final_val("std_L2")
    _, vk = final_val("flynetS_L2k25")
    check("C2-334M-b8-k25-wins", vk["val"] < vd["val"],
          f"dense {vd['val']} vs k25 {vk['val']}")

    # ---- C3: 478M b4 12k: confounded pair, k25 loses (must NOT be claimed as win)
    _, vd = final_val("std_XL")
    _, vk = final_val("flynetS_XLk25")
    check("C3-478M-b4-12k-dense-wins", vk["val"] > vd["val"],
          f"dense {vd['val']} vs k25 {vk['val']} (confounded pair, excluded)")

    # ---- C4: 478M b4 24k dense final (k25 24k rerun pending/completed separately)
    _, vd = final_val("std_XL")
    check("C4-478M-24k-dense-final", abs(vd["val"] - 5.4736) < 1e-6,
          f"dense 24k final {vd['val']}")

    # ---- C5: 0.5B matched pair at b4 (12.3M tok): k25 wins
    _, vd = final_val("std_XLb4")
    _, vk = final_val("flynetS_XLb4k25")
    check("C5-478M-b4-12k-k25-wins", vk["val"] < vd["val"],
          f"dense {vd['val']} vs k25 {vk['val']}")

    # ---- C6: 0.5B budget doubling: 12k pair (both b4/12.3M tok) vs 24k pair
    _, dd12 = final_val("std_XLb4")
    _, dk12 = final_val("flynetS_XLb4k25")
    gap12 = dd12["val"] - dk12["val"]     # 12k pair, both arms same schedule
    gap24 = 5.4861 - 5.3059               # 24k pair (from probes below)
    pk = [json.loads(l) for l in
          open(os.path.join(L, "flynetS_XL24k25_probes.jsonl"), encoding="utf-8")]
    pd = [json.loads(l) for l in
          open(os.path.join(L, "std_XL24_probes.jsonl"), encoding="utf-8")]
    k24 = [d for d in pk if d["step"] == 24000][0]["val"]
    d24 = [d for d in pd if d["step"] == 24000][0]["val"]
    gap24 = d24 - k24
    check("C6-478M-budget-doubling-grows-gap", gap24 > gap12 > 0,
          f"12k gap {gap12:+.4f} < 24k gap {gap24:+.4f} (training-budget scaling)")

    # ---- C7: U-curve minimum at 25% (9-point sweep finals)
    sweep = {}
    for k, tag in [(2, "flynetS_k02"), (5, "flynetS_k05"), (15, "flynetS_k15"),
                   (25, "flynetS_k25"), (40, "flynetS_k40"), (50, "flynetS_k50"),
                   (60, "flynetS_k60")]:
        try:
            sweep[k] = final_val(tag)[1]["val"]
        except FileNotFoundError:
            pass
    # dense 12k anchor: mean of the two committed 12k dense seeds
    _, s8 = final_val("std_s8")
    _, s9 = final_val("std_s9")
    sweep[100] = round((s8["val"] + s9["val"]) / 2, 4)   # dense anchor (12k sched)
    check("C7-sweep-min-at-25", min(sweep, key=sweep.get) == 25,
          f"argmin={min(sweep, key=sweep.get)}% vals={sweep} dense-anchor {sweep[100]}")

    # ---- C8: throttle verdict — random-skip within 0.03 of gate on both metrics
    g = load("cl_fly_std_s8_result.json")   # 4-domain gate, seed8 (s7 json
    sk = load("cl_skip_std_result.json")    # was overwritten by the 8-domain run
    ok = (abs(sk["avg_forgetting"] - g["avg_forgetting"]) < 0.03 and
          abs(sk["avg_improvement"] - g["avg_improvement"]) < 0.03)
    check("C8-random-skip-matches-gate", ok,
          f"skip(s7) {sk['avg_forgetting']}/{sk['avg_improvement']} vs "
          f"gate(s8) {g['avg_forgetting']}/{g['avg_improvement']} (cross-seed)")

    # ---- C9: causal ablation — code channels >= 1.5x random
    tr = load("code_transplant_result.json")
    ratio = tr["zero_top25"]["kl"] / max(tr["zero_random25"]["kl"], 1e-9)
    check("C9-causal-ablation-1.5x", ratio >= 1.5,
          f"zero_top25 {tr['zero_top25']['kl']} vs zero_random25 "
          f"{tr['zero_random25']['kl']} ratio {ratio:.2f}")

    # ---- C10: three-way shuffle ordering own > mismatch > random
    sh = load("code_shuffle_result.json")
    ok = (sh["zero_own"]["kl"] > sh["zero_mismatch"]["kl"] >
          sh["zero_random"]["kl"])
    check("C10-shuffle-ordering", ok,
          f"own {sh['zero_own']['kl']} > mismatch {sh['zero_mismatch']['kl']} "
          f"> random {sh['zero_random']['kl']}")

    # ---- C11: topic retrieval — hamming >= 2x chance at hit@1
    dr = load("domain_retrieval_result.json")
    check("C11-topic-retrieval-2x", dr["hamming_topic_hit1"] >= 0.25,
          f"hamming hit@1 {dr['hamming_topic_hit1']} (chance 0.125)")

    # ---- C12: store scaling — hit@10 >= 0.75 at 40k store
    check("C12-store-scaling-hit10", dr["hamming_topic_hit10"] >= 0.75,
          f"40k-store hit@10 {dr['hamming_topic_hit10']}")

    # ---- C13: developmental freezing — monotone rise 0.45 -> 0.89
    dc = load("dev_crosssnap_result.json")
    steps = sorted(int(k) for k in dc if k != "chance_same_snapshot")
    vals = [dc[str(st)] for st in steps]
    mono = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    check("C13-dev-freezing-monotone", mono and vals[-1] >= 0.85,
          f"{list(zip(steps[1:], vals))} chance {dc['chance_same_snapshot']}")

    # ---- C14: elite census — all ginis >= 0.45
    ec = load("elite_census_result.json")
    ginis = {k: v["gini_mean"] for k, v in ec.items() if isinstance(v, dict)}
    check("C14-elite-gini-range", all(v >= 0.45 for v in ginis.values()),
          f"{ginis}")

    # ---- summary
    n_pass = sum(1 for c in checks if c["pass"])
    print(f"\n{n_pass}/{len(checks)} checks PASS", flush=True)
    with open(os.path.join(L, "credentials_report.json"), "w") as f:
        json.dump({"pass": n_pass, "total": len(checks), "checks": checks}, f,
                  indent=1)
    print("saved logs_v2/credentials_report.json", flush=True)


if __name__ == "__main__":
    main()
