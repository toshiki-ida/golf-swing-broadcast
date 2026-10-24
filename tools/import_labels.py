"""import_labels — tests/club_labels.json (label_club.py の正解ラベル) を datasets/clubhead の形式に取り込む。

    python tools/import_labels.py [--dataset <保存先>]   (既定: リポジトリ直下の datasets/clubhead)

各動画のラベル点を manual (labeler は club_labels.json の labeler: owner / claude) として保存する。
動画は labels.json の "video" (リポジトリ相対) → 無ければ同名ファイルを tests/real/trimmed, tests/real,
build/club_eval/src から探す。holdout のクリップは除外。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.dataset import collect_clip_labels, is_holdout, summary  # noqa: E402
from ai.config import realtime_factor  # noqa: E402
from clip_manager import TrajectoryData  # noqa: E402

SEARCH = [ROOT / "tests" / "real" / "trimmed", ROOT / "tests" / "real", ROOT / "build" / "club_eval" / "src",
          Path(r"C:/Users/yoshi/Desktop/swingcut/output/sample")]


def find_video(name: str, hint: str) -> Path | None:
    cands = [ROOT / hint] if hint else []
    cands += [d / name for d in SEARCH]
    for c in cands:
        if c.exists():
            return c
    return None


def main() -> int:
    import cv2
    root = Path(sys.argv[sys.argv.index("--dataset") + 1]) if "--dataset" in sys.argv else None
    labels = json.loads((ROOT / "tests" / "club_labels.json").read_text(encoding="utf-8"))
    n_done = 0
    for name, lab in labels.items():
        frames = lab.get("frames") or {}
        pts = [(v["x"], v["y"], int(f)) for f, v in frames.items() if v.get("x") is not None]
        if len(pts) < 2:
            print(f"[skip] {name}: 座標付きの点が 2 点未満")
            continue
        video = find_video(name, lab.get("video", ""))
        if video is None:
            print(f"[skip] {name}: 動画が見つかりません")
            continue
        if is_holdout(video):
            print(f"[skip] {name}: holdout")
            continue
        fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
        real_fps = fps * realtime_factor(fps, "auto")
        sw = TrajectoryData(points=sorted(pts, key=lambda p: p[2]))
        res = collect_clip_labels(f"label_{Path(name).stem}", video, 0, [sw], corrections=None, landmarks=None,
                                  body_h=lab.get("height", 1080) * 0.35, real_fps=real_fps,
                                  labeler=lab.get("labeler", "owner"), note="tests/club_labels.json から取り込み",
                                  out_root=root)
        print(f"[ok  ] {name}: {res}")
        n_done += 1
    print(f"取り込み {n_done} 本 → {summary(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
