"""build_prior — datasets/clubhead の承認済み軌道から、推定用テンプレート models/trajectory_prior_<angle>.json を作る。

    python tools/build_prior.py [--dataset <保存先>] [--out models]

撮影方向 (labels.json の camera) ごとに、手首のアドレスからの相対位置・手首の速度の向き・ヘッドの手首からのオフセットを
身長代替値で正規化して貯める。レコードに wrist が無ければ切り出し画像に MediaPipe を掛けて求め、labels.json に書き戻す。
サンプルが MIN_PRIOR_POINTS (150) 未満の方向は「テンプレート無し」(物理制約のみ) として扱われる。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.dataset import DATASET_DIR  # noqa: E402
from ai.trajectory_prior import build_priors, MIN_PRIOR_POINTS  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    root = Path(args[args.index("--dataset") + 1]) if "--dataset" in args else DATASET_DIR
    out = Path(args[args.index("--out") + 1]) if "--out" in args else None
    summary = build_priors(root, out)
    for angle, s in summary.items():
        print(f"{angle:<8} クリップ {s['n_clips']:>3} 本 / サンプル {s['n_points']:>5} 点 / クラブ長 {s['club_len']:.2f} 身長  "
              f"{'テンプレートとして使う' if s['enough'] else f'不足 (< {MIN_PRIOR_POINTS}) → 物理制約のみ'}")
    if not summary:
        print("テンプレートを作れるクリップがありません (camera が dtl / face_on のクリップが必要)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
