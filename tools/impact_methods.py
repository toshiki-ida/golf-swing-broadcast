"""impact_methods — 骨格由来のインパクト検出の改善案を並べて比較する (club_tracker / analyzer は変更しない)。

    python tools/impact_methods.py --split tune [--ref claude|owner]

build/club_eval/real/<名前>/club_track.json (骨格 33 点・全フレーム) と club_tracker.pred.json を読み、
各素材で以下の候補フレームを計算して一覧にする。

  wrist_near   : 現行。トップ後に生の手首中点がアドレス時の位置へ最接近する点
  wrist_speed  : 最接近の直前 (窓 = 最接近 − 0.25 秒 〜 最接近) で手首速度が最大となる点 ※案 1
  wrist_lowest : トップ後に手首中点が最も低く (y 最大) なる点 ※案 2
  hands_ahead  : 手首 x がアドレス時の手首 x を (ターゲット方向に) 越えた最初の点 ※案 3
  traj_near    : クラブヘッド軌道 (全フレーム点) がアドレス時のヘッド位置へ最初に戻る点 (sns_render.derive_phases)
  ball_roi     : ボール ROI 差分 (analyzer の結果)

基準 (--ref): tests/club_labels.json の impact ラベル (labeler で選別)。無ければ traj_near を参考値にする。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):   # Windows の cp932 コンソールでも落ちないように
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.config import realtime_factor  # noqa: E402
from ai.store import landmarks_from_track  # noqa: E402
from ai.swing_analyzer import (L_WRIST, R_WRIST, build_tracks, detect_phases_wrist, smooth)  # noqa: E402
from ai.sns_render import derive_phases  # noqa: E402
from clip_manager import TrajectoryData  # noqa: E402

REAL_DIR = ROOT / "tests" / "real"


def methods_for(video: Path, out_dir: Path) -> dict:
    import cv2
    import yaml  # noqa: F401
    track = json.loads((out_dir / "club_track.json").read_text(encoding="utf-8"))
    pred = json.loads((out_dir / "club_tracker.pred.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.release()
    real_fps = fps * realtime_factor(fps, "auto")
    scale = max(1.0, real_fps / 30.0)
    lm = landmarks_from_track(track)
    frames, raw = build_tracks(lm, W, H)
    tracks = smooth(raw, scale)
    ph = detect_phases_wrist(tracks, raw, real_fps, scale)
    top, cand = ph["top"], ph["candidate"]
    wx = (tracks[L_WRIST][:, 0] + tracks[R_WRIST][:, 0]) / 2
    wy = (tracks[L_WRIST][:, 1] + tracks[R_WRIST][:, 1]) / 2
    speed = ph["speed"]
    out = {"wrist_near": cand}
    # 案 1: 最接近の直前 0.25 秒で速度最大
    w = max(2, int(round(0.25 * real_fps)))
    lo = max(top + 1, cand - w)
    out["wrist_speed"] = int(lo + np.argmax(speed[lo:cand + 1])) if cand > lo else cand
    # 案 2: トップ後、手首が最も低い点 (インパクト前後 0.5 秒以内に限定)
    hi = min(len(wy) - 1, cand + int(round(0.5 * real_fps)))
    out["wrist_lowest"] = int(top + 1 + np.argmax(wy[top + 1:hi + 1])) if hi > top + 1 else cand
    # 案 3: 手首 x がアドレス時の x を越えた最初の点 (ハンドファースト)。方向はトップ→候補の x 変化から
    addr = ph["address"]
    direction = np.sign(wx[cand] - wx[top]) or 1.0
    crossed = [i for i in range(top + 1, hi + 1) if (wx[i] - wx[addr]) * direction > 0]
    out["hands_ahead"] = int(crossed[0]) if crossed else cand
    # クラブヘッド軌道由来
    dense = [tuple(p) for p in track["points"]]
    swings = [TrajectoryData(points=[tuple(p) for p in track["confident_points"]])]
    try:
        out["traj_near"] = derive_phases(swings, 0, dense)["impact"]
    except Exception:
        out["traj_near"] = None
    out["_meta"] = {"fps": fps, "real_fps": real_fps, "top": top, "frames": frames}
    return out


def main() -> int:
    import yaml
    args = sys.argv[1:]
    split = args[args.index("--split") + 1] if "--split" in args else "tune"
    ref_kind = args[args.index("--ref") + 1] if "--ref" in args else "owner"
    labels = json.loads((ROOT / "tests" / "club_labels.json").read_text(encoding="utf-8"))
    sp = yaml.safe_load((REAL_DIR / "split.yaml").read_text(encoding="utf-8"))
    names = list(sp.get(split, {}))
    src_dir = REAL_DIR / "trimmed"
    cols = ["wrist_near", "wrist_speed", "wrist_lowest", "hands_ahead", "traj_near"]
    print(f"{'素材':<22} {'基準':>6}  " + "".join(f"{c:>13}" for c in cols) + "   (値 = フレーム, 括弧 = 基準との差を実時間 ms で)")
    errs = {c: [] for c in cols}
    for n in names:
        stem = Path(n).stem
        video = src_dir / f"{stem}_trim.mp4"
        out_dir = ROOT / "build" / "club_eval" / "real" / f"{stem}_trim"
        if not (out_dir / "club_track.json").exists():
            print(f"{stem:<22} 追跡結果なし")
            continue
        try:
            m = methods_for(video, out_dir)
        except Exception as e:
            print(f"{stem:<22} エラー: {type(e).__name__}: {e}")
            continue
        real_fps = m["_meta"]["real_fps"]
        lab = labels.get(video.name) or labels.get(n) or {}
        ref = None
        if lab and lab.get("labeler", "claude") == ref_kind:
            imp = [int(f) for f, v in lab.get("frames", {}).items() if v.get("phase") == "impact"]
            ref = imp[0] if imp else None
        line = f"{stem:<22} {str(ref) if ref is not None else '-':>6}  "
        for c in cols:
            v = m[c]
            if v is None:
                line += f"{'-':>13}"
                continue
            if ref is not None:
                d = (v - ref) / real_fps * 1000
                errs[c].append(abs(d))
                line += f"{v:>6} ({d:+5.0f})"
            else:
                line += f"{v:>13}"
        print(line)
    if any(errs.values()):
        print()
        for c in cols:
            if errs[c]:
                print(f"{c:<13}: 平均 |誤差| {np.mean(errs[c]):6.0f} ms  最大 {np.max(errs[c]):6.0f} ms  (n={len(errs[c])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
