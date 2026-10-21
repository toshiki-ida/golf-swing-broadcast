"""trim_real — tests/real/trim.yaml に従って素材を「スイング 1 回分」に切り出す。

    python tools/trim_real.py [--force]

出力: tests/real/trimmed/<名前>_trim.mp4 (git 管理外)。ffmpeg でフレーム精度に再エンコード (libx264 CRF 18)。
HFR 2x 素材はフレームをそのまま残す (fps は変えない)。skip: true の素材は作らない。
既に存在する出力は --force が無ければ飛ばす。
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):   # Windows の cp932 コンソールでも落ちないように
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ffmpeg_writer import ffmpeg_extract  # noqa: E402

REAL_DIR = ROOT / "tests" / "real"
TRIM_DIR = REAL_DIR / "trimmed"


def main() -> int:
    force = "--force" in sys.argv[1:]
    spec = yaml.safe_load((REAL_DIR / "trim.yaml").read_text(encoding="utf-8")) or {}
    TRIM_DIR.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for name, cfg in spec.items():
        src = REAL_DIR / name
        if cfg.get("skip"):
            print(f"[skip] {name}: {cfg.get('note', '')}")
            skipped += 1
            continue
        if not src.exists():
            print(f"[miss] {name}: 元素材がありません")
            continue
        dst = TRIM_DIR / f"{src.stem}_trim.mp4"
        if dst.exists() and not force:
            print(f"[keep] {dst.name}")
            done += 1
            continue
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        start = int(cfg.get("start", 0))
        end = min(int(cfg.get("end", total - 1)), total - 1)
        ok = ffmpeg_extract(str(src), str(dst), start, end, fps, crf=18, preset="medium", hw_encode=False)
        if ok:
            cap = cv2.VideoCapture(str(dst))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"[ok  ] {dst.name}: {start}-{end} ({end - start + 1} 要求 / {n} 出力) fps={fps:.2f}")
            done += 1
        else:
            print(f"[fail] {name}")
    print(f"完了: {done} 本, skip {skipped} 本 → {TRIM_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
