"""setup_assets.py — AI 解析に必要なアセットを取得する(ネット接続時に 1 回だけ実行)。

    python setup_assets.py

取得するもの:
  - models/pose_landmarker_heavy.task  (MediaPipe Pose Landmarker heavy モデル, 約 30 MB)
  - assets/fonts/NotoSansJP.ttf        (日本語フォント / SIL Open Font License, 約 10 MB)

どちらも .gitignore 済み。PyInstaller の spec はこれらが存在すれば exe に同梱する。
アプリ本体の実行パスからこのスクリプトを呼んではならない(実行時はオフライン動作)。
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ASSETS = [
    {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
        "dest": ROOT / "models" / "pose_landmarker_heavy.task",
    },
    {
        "url": (
            "https://raw.githubusercontent.com/google/fonts/main/"
            "ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
        ),
        "dest": ROOT / "assets" / "fonts" / "NotoSansJP.ttf",
    },
]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.relative_to(ROOT)} (already exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 16):
                f.write(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [ ok ] {dest.relative_to(ROOT)} ({size_mb:.1f} MB)")


def main() -> int:
    print("Golf Swing Broadcast asset setup")
    ok = True
    for asset in ASSETS:
        try:
            download(asset["url"], asset["dest"])
        except Exception as e:
            ok = False
            print(f"  [FAIL] {asset['dest'].name}: {e}", file=sys.stderr)
    print("done" if ok else "finished with errors")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
