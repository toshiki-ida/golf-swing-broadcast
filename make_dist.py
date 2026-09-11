"""配布フォルダを組み立てる

PyInstaller の COLLECT は dist/GolfSwingBroadcast を毎回作り直すので、
モデルは手で入れ直すことになる。取りこぼすと配布先で初めて気付くため、
ここに手順を固定する。

    python -m PyInstaller GolfSwingBroadcast.spec --noconfirm
    python make_dist.py

モデルの置き場所:
  yolo11m-seg.pt        exe の隣 (ultralytics が CWD 基準で探すため)
  models/rife/          _internal の下 (凍結時 __file__ が _internal を指す)

配布前チェックも兼ねる。settings.json / clips.json が残っていると、
このPCのユーザー名を含むパスが配布先に持ち込まれ、起動時の mkdir が
PermissionError で落ちる。
"""

import shutil
import sys
from pathlib import Path

SRC = Path(__file__).parent
DIST = SRC / "dist" / "GolfSwingBroadcast"

# exe の隣に置くもの
BESIDE_EXE = ["yolo11m-seg.pt"]

# _internal の下に置くもの (SRC からの相対パス)
INTERNAL = [
    "models/rife/rife425_fp16.engine",
    "models/rife/train_log/flownet.pkl",
    "models/rife/train_log/IFNet_HDv3.py",
    "models/rife/train_log/RIFE_HDv3.py",
    "models/rife/train_log/__init__.py",
    "models/rife/model/__init__.py",
    "models/rife/model/warplayer.py",
]

# 実行時には要らない。onnx は 300MB 超あるので必ず除く
NEVER_SHIP = ["*.onnx", "settings.json", "clips.json", "playout.json"]

# ffmpeg。spec の binaries に入れても取りこぼすことがあり、無いと編集タブの
# 書き出しが cv2 フォールバックに落ちる。ffmpeg.exe は共有ビルドなので
# 同じ場所の DLL も一緒に要る。
FFMPEG_BIN = Path(r"C:\ffmpeg\bin")


def ship_ffmpeg():
    """ffmpeg.exe と依存DLLを _internal に置く (find_ffmpeg が最初に見る場所)"""
    dst_dir = DIST / "_internal"
    if (dst_dir / "ffmpeg.exe").exists():
        return 0
    src_exe = FFMPEG_BIN / "ffmpeg.exe"
    if not src_exe.exists():
        sys.exit(f"ffmpeg がありません: {src_exe}\n"
                 "無いまま配布すると書き出しが cv2 フォールバックに落ちます。")
    n = 0
    for src in [src_exe] + sorted(FFMPEG_BIN.glob("*.dll")):
        dst = dst_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        n += 1
    return n


def main():
    if not (DIST / "GolfSwingBroadcast.exe").exists():
        sys.exit(f"exe がありません。先に PyInstaller を実行してください: {DIST}")

    n = ship_ffmpeg()
    for rel in BESIDE_EXE:
        src = SRC / rel
        if not src.exists():
            sys.exit(f"見つかりません: {src}")
        shutil.copy2(src, DIST / src.name)
        n += 1

    for rel in INTERNAL:
        src = SRC / rel
        if not src.exists():
            sys.exit(f"見つかりません: {src}")
        dst = DIST / "_internal" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1

    # 前回の動作確認で作られた設定・作業ファイルを落とす
    removed = []
    for pat in NEVER_SHIP:
        for p in DIST.rglob(pat):
            p.unlink()
            removed.append(p.relative_to(DIST))
    for d in DIST.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)

    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"同梱 {n} ファイル")
    for r in removed:
        print(f"除去 {r}")
    print(f"{DIST}  {size / 1e9:.2f} GB")
    print("このフォルダをそのまま配布できます")


if __name__ == "__main__":
    main()
