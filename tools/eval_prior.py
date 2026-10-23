"""eval_prior — 推定 (trajectory_prior) の精度を、観測を隠して測る。

    python tools/eval_prior.py --split holdout|tune|all [--hide 0.3] [--block-sec 0.25] [--seed 0] [--tag <tag>]

holdout には正解ラベルが無いので、信頼できる観測区間の一部を隠し (block-sec 秒のブロックを hide の割合)、
隠したフレームを prior で推定して観測との px 誤差を測る。観測区間 (隠していないフレーム) は観測そのものなので誤差 0 だが、
参考として「観測の 1 フレーム前後のばらつき」ではなく、隠さなかった区間の長さだけ出す。
prior あり / prior なし (物理制約のみ) の両方を同じ表に出し、撮影方向と局面 (アドレス〜トップ / トップ〜インパクト / インパクト後)
ごとに分ける。入力は build/club_eval/real/<stem>/club_track.json (tools/eval_club.py の出力。手首・骨格入り)。
結果は tests/real/results/<日時>_prior_<tag>_<split>.md。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.club_tracker import ClubTrackResult, resolve_gate  # noqa: E402
from ai.trajectory_prior import fill_trajectory, load_prior, swing_frame_range, estimate_camera_from_landmarks  # noqa: E402

RUNS = ROOT / "build" / "club_eval" / "real"


def load_track(stem: str):
    p = RUNS / stem / "club_track.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    res = ClubTrackResult(points=[tuple(p_) for p_ in d["points"]], conf={int(k): v for k, v in d["conf"].items()},
                          wrist={int(k): tuple(v) for k, v in d.get("wrist", {}).items()},
                          body_h=float(d["body_h"]), real_fps=float(d.get("real_fps", 30.0)))
    res.attached = {int(f): True for f in d.get("attached", [])}
    res.landmarks = {int(k): [tuple(xy) for xy in v] for k, v in d.get("landmarks", {}).items()}
    return res, d


def phase_of(f, rng, obs_frames):
    """局面の粗い分類: アドレス〜トップ (前半 40%) / ダウン (40〜60%) / インパクト後"""
    a, b = rng
    t = (f - a) / max(b - a, 1)
    return "address-top" if t < 0.4 else "down" if t < 0.6 else "follow"


def main() -> int:
    args = sys.argv[1:]
    split = args[args.index("--split") + 1] if "--split" in args else "holdout"
    hide = float(args[args.index("--hide") + 1]) if "--hide" in args else 0.3
    block_sec = float(args[args.index("--block-sec") + 1]) if "--block-sec" in args else 0.25
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
    tag = args[args.index("--tag") + 1] if "--tag" in args else "prior"
    sp = yaml.safe_load((ROOT / "tests" / "real" / "split.yaml").read_text(encoding="utf-8"))
    stems = []
    for part in (["tune", "holdout"] if split == "all" else [split]):
        stems += [Path(k).stem + "_trim" for k in (sp.get(part) or {})]
    rng = random.Random(seed)
    rows = []
    agg = {}   # (prior?, angle, phase) -> errors
    for stem in stems:
        got = load_track(stem)
        if got is None:
            print(f"[skip] {stem}: club_track.json がありません (tools/eval_club.py --all --split {split} を先に)")
            continue
        res, d = got
        W = H = None
        if res.landmarks:
            xs = [p[0] for lm in res.landmarks.values() for p in lm]
            ys = [p[1] for lm in res.landmarks.values() for p in lm]
            W, H = (1920, 1080) if max(xs) > 1100 or max(ys) > 1100 else (1280, 720)
        angle = estimate_camera_from_landmarks(res.landmarks, W or 1920, H or 1080, res.real_fps)
        prior = load_prior(angle) if angle in ("dtl", "face_on") else None
        good = [seg for seg, q, ok in res.segment_quality(gate=resolve_gate()) if ok]
        obs = {int(p[2]): (p[0], p[1]) for seg in good for p in seg}
        if len(obs) < 20:
            rows.append((stem, angle, 0, 0, None, None, "観測が少ない"))
            continue
        frames = sorted(obs)
        block = max(2, int(round(block_sec * res.real_fps)))
        # ブロック単位で hide の割合を隠す (両端のブロックは残す)
        blocks = [frames[i:i + block] for i in range(0, len(frames), block)]
        inner = list(range(1, len(blocks) - 1))
        rng.shuffle(inner)
        hidden = set()
        for bi in inner[:int(round(len(inner) * hide))]:
            hidden.update(blocks[bi])
        visible = {f: obs[f] for f in frames if f not in hidden}
        swing_rng = swing_frame_range(res.landmarks, W or 1920, H or 1080, res.real_fps) or (frames[0], frames[-1])
        for use_prior in (True, False):
            filled = fill_trajectory(visible, res.wrist, res.body_h, res.real_fps, frames, prior=prior if use_prior else None)
            est = {f: (x, y) for x, y, f, s in filled if s == "estimated"}
            errs = []
            for f in sorted(hidden):
                if f in est:
                    e = float(np.hypot(est[f][0] - obs[f][0], est[f][1] - obs[f][1]))
                    errs.append(e)
                    agg.setdefault((use_prior, angle, phase_of(f, swing_rng, frames)), []).append(e / res.body_h)
            if errs:
                rows.append((stem, angle, len(hidden), len(visible), round(float(np.median(errs)), 1), round(float(np.mean(errs)), 1),
                             ("prior" if use_prior else "物理のみ") + ("" if (prior is not None or not use_prior) else " (テンプレート無し→物理)")))
    lines = [f"# 推定 (trajectory_prior) 評価 ({time.strftime('%Y%m%d_%H%M')}, tag={tag}, split={split}, hide={hide}, block={block_sec}s, seed={seed})", "",
             "観測区間の一部をブロック単位で隠し、隠したフレームを推定して観測との px 誤差 (元解像度) を測る。観測区間は観測そのものを使うので誤差 0。", "",
             f"{'素材':<26} {'方向':<8} {'隠した':>5} {'残した':>5} {'中央値px':>8} {'平均px':>7}  方式"]
    for r in rows:
        lines.append(f"{r[0]:<26} {r[1]:<8} {r[2]:>5} {r[3]:>5} {str(r[4]):>8} {str(r[5]):>7}  {r[6]}")
    lines += ["", "## 撮影方向 × 局面 (誤差 / 身長代替値、中央値)", "", f"{'方式':<8} {'方向':<8} {'局面':<12} {'n':>5} {'中央値':>7} {'平均':>7}"]
    for (use_prior, angle, ph), es in sorted(agg.items(), key=lambda kv: (not kv[0][0], kv[0][1], kv[0][2])):
        a = np.array(es)
        lines.append(f"{'prior' if use_prior else '物理':<8} {angle:<8} {ph:<12} {len(a):>5} {np.median(a):>7.3f} {a.mean():>7.3f}")
    text = "\n".join(lines) + "\n"
    print(text)
    out = ROOT / "tests" / "real" / "results" / f"{time.strftime('%Y%m%d_%H%M')}_prior_{tag}_{split}.md"
    out.write_text(text, encoding="utf-8")
    print(f"結果を保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
