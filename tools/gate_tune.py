"""gate_tune — 信頼度ゲートの閾値を、区間ごとの目視判定 (tests/real/segment_review.yaml) と突き合わせて決める。

    python tools/gate_tune.py [--grid] [--gate key=value ...]

- 入力: build/club_eval/real/<stem>/club_track.json (tools/run_club_tracker.py が保存。手首・接続フラグ・骨格入り)
        tests/real/segment_review.yaml (区間の frames と verdict: good / bad / mixed)
- 出力: 区間ごとの指標 (観測・率・連続・接続・動き・頭上) と判定、閾値候補ごとの「bad を通した数 / good を通した数」
- 既定の閾値は ai.config.gate_thresholds() (config/ai.json)。--gate で上書き、--grid で格子探索。
mixed は「描かない側」に倒す (bad と同じ扱い)。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import itertools
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.club_tracker import ClubTrackResult, resolve_gate, DEFAULT_GATE  # noqa: E402

REVIEW = ROOT / "tests" / "real" / "segment_review.yaml"
RUNS = ROOT / "build" / "club_eval" / "real"


def load_result(stem: str) -> ClubTrackResult | None:
    p = RUNS / stem / "club_track.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    res = ClubTrackResult(points=[tuple(p_) for p_ in d["points"]], conf={int(k): v for k, v in d["conf"].items()},
                          wrist={int(k): tuple(v) for k, v in d.get("wrist", {}).items()},
                          body_h=float(d["body_h"]), real_fps=float(d.get("real_fps", 30.0)))
    res.attached = {int(f): True for f in d.get("attached", [])}
    res.landmarks = {int(k): [tuple(xy) for xy in v] for k, v in d.get("landmarks", {}).items()}
    return res


def match(frames, review_rows):
    """区間の frames に一番重なる判定行を返す"""
    a, b = frames
    best, best_ov = None, 0
    for r in review_rows:
        ra, rb = r["frames"]
        ov = max(0, min(b, rb) - max(a, ra) + 1)
        if ov > best_ov:
            best, best_ov = r, ov
    return best if best_ov > 0 else None


def evaluate(gate: dict, results: dict, review: dict, verbose: bool = False):
    """gate で各区間を判定し、(bad/mixed を通した数, good を通した数, good の総数, bad の総数, 行) を返す"""
    bad_pass = good_pass = n_good = n_bad = 0
    rows = []
    for stem, res in results.items():
        for seg, q, ok in res.segment_quality(gate=gate):
            r = match(q["frames"], review.get(stem, []))
            verdict = (r or {}).get("verdict", "?")
            is_good = verdict == "good"
            if verdict == "?":
                continue
            n_good += is_good
            n_bad += not is_good
            if ok and is_good:
                good_pass += 1
            if ok and not is_good:
                bad_pass += 1
            rows.append((stem, q, ok, verdict))
    return bad_pass, good_pass, n_good, n_bad, rows


def main() -> int:
    args = sys.argv[1:]
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8")) or {}
    results = {}
    for stem in review:
        res = load_result(stem)
        if res is None:
            print(f"[skip] {stem}: club_track.json がありません (tools/eval_club.py --all --split tune を先に)")
            continue
        results[stem] = res
    gate = resolve_gate()
    if "--gate" in args:
        for kv in args[args.index("--gate") + 1:]:
            if "=" not in kv:
                break
            k, v = kv.split("=", 1)
            gate[k] = float(v) if "." in v else int(v)

    bad_pass, good_pass, n_good, n_bad, rows = evaluate(gate, results, review)
    print(f"閾値: {gate}")
    print(f"{'区間':<40} {'観測':>4} {'率':>5} {'連続':>5} {'接続':>5} {'動き':>5} {'頭上':>5} {'距σ':>5} {'距Δ':>6} {'速比':>5} {'手':>5}  判定  目視  理由")
    for stem, q, ok, verdict in rows:
        a, b = q["frames"]
        mark = "OK" if ok else "--"
        flag = "" if (ok == (verdict == "good")) else "  <-- 不一致"
        print(f"{stem[9:]:<26} F{a:>3}-{b:<4}     {q['observed']:>4} {q['rate']:>5.2f} {q['continuity']:>5.2f} {q['attached']:>5.2f} "
              f"{q['motion_bh']:>5.2f} {q['on_head']:>5.2f} {q['dist_std']:>5.2f} {q['dist_change']:>6.3f} "
              f"{min(q['speed_ratio'], 99):>5.2f} {q['wrist_motion']:>5.2f}  {mark:<4}  {verdict:<5} {' / '.join(q['reasons'])}{flag}")
    print(f"\n通過: good {good_pass}/{n_good}, bad/mixed {bad_pass}/{n_bad}")

    if "--grid" in args:
        print("\n=== 格子探索 (bad/mixed の通過が少なく、good の通過が多い順) ===")
        grid = {
            "min_observed": [6, 8, 10],
            "min_hand_connect_ratio": [0.0, 0.05, 0.10],
            "min_motion_bh": [0.0, 0.3, 0.5],
            "max_on_head_ratio": [0.3, 1.0],
            "max_dist_std": [0.15, 0.19, 0.22, 9.0],
            "max_dist_change": [0.06, 0.07, 0.08, 9.0],
            "min_speed_ratio": [0.0, 1.0, 1.2],
            "min_wrist_ratio": [0.0, 0.05],
        }
        keys = list(grid)
        scored = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            g = dict(gate)
            g.update(dict(zip(keys, combo)))
            bp, gp, ng, nb, _ = evaluate(g, results, review)
            scored.append((bp, -gp, combo))
        scored.sort()
        for bp, ngp, combo in scored[:15]:
            print(f"  bad通過 {bp:>2}  good通過 {-ngp:>2}/{n_good}  " + ", ".join(f"{k}={v}" for k, v in zip(keys, combo)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
