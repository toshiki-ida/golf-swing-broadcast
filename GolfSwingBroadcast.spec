# -*- mode: python ; coding: utf-8 -*-
import glob, importlib, os, shutil
from PyInstaller.utils.hooks import collect_all
_ctk_dir = os.path.dirname(importlib.import_module('customtkinter').__file__)

# Collect all submodules, binaries and data for numpy/scipy 2.x compatibility
_np_datas, _np_binaries, _np_hiddenimports = collect_all('numpy')
_sp_datas, _sp_binaries, _sp_hiddenimports = collect_all('scipy')

# MediaPipe (AI 解析用)。tasks API は .binarypb / .so 等のデータファイルを
# パッケージ内から読むため collect_all で丸ごと同梱する。
# app.py は mediapipe を直接 import しないので hiddenimports でも明示する。
# matplotlib (約 15 MB) は mediapipe.tasks.python.vision.__init__ が無条件に import する
# ため除外できない (除外すると frozen 環境で `from mediapipe.tasks.python import vision` が失敗する)。
_mp_datas, _mp_binaries, _mp_hiddenimports = collect_all('mediapipe')
# onnxruntime (クラブヘッド検出器 models/clubhead.onnx の推論)。DLL を含むので丸ごと同梱する
_ort_datas, _ort_binaries, _ort_hiddenimports = collect_all('onnxruntime')
_mp_hiddenimports += ['mediapipe', 'mediapipe.tasks.python',
                      'mediapipe.tasks.python.vision']

# ai パッケージは app.py から関数内 import (try/except) で読まれるため明示する
_ai_hiddenimports = ['ai', 'ai.config', 'ai.store', 'ai.edit_hook', 'ai.club_tracker', 'ai.club_detector', 'ai.dataset',
                     'ai.swing_analyzer', 'ai.sns_render', 'ai.selftest',
                     'yaml', 'scipy.signal', 'scipy.interpolate', 'PIL.ImageFont', 'PIL.ImageDraw']

# AI アセット (setup_assets.py で取得)。存在するものだけ同梱する。
# config/ (rules.yaml) は _internal/config に入れる。exe 同階層に config/ を置けばそちらが優先される
# (ai.swing_analyzer.rules_path 参照)。
_asset_datas = []
for _src, _dst in [('models', 'models'), (os.path.join('assets', 'fonts'), os.path.join('assets', 'fonts')),
                   ('config', 'config')]:
    if os.path.isdir(_src):
        _asset_datas.append((_src, _dst))

# ffmpeg / ffprobe 同梱 (EXEフォルダだけで動作するように)
#   1. C:\ffmpeg\bin (従来の固定パス)
#   2. PATH 上の ffmpeg (winget の Gyan.FFmpeg はシンボリックリンクなので実体に辿る)
# どちらにも無ければ同梱しない (録画・書き出しは実行時に PATH の ffmpeg を探す)。
def _find_ffmpeg_dir():
    fixed = r'C:\ffmpeg\bin'
    if os.path.exists(os.path.join(fixed, 'ffmpeg.exe')):
        return fixed
    exe = shutil.which('ffmpeg')
    if exe:
        return os.path.dirname(os.path.realpath(exe))
    return None

_ffmpeg_dir = _find_ffmpeg_dir()
_ffmpeg_files = []
if _ffmpeg_dir:
    _ffmpeg_files = [(f, '.') for f in
                     glob.glob(os.path.join(_ffmpeg_dir, 'ffmpeg.exe')) +
                     glob.glob(os.path.join(_ffmpeg_dir, 'ffprobe.exe')) +
                     glob.glob(os.path.join(_ffmpeg_dir, '*.dll'))
                     if os.path.exists(f)]
    print(f'[spec] ffmpeg: {_ffmpeg_dir} ({len(_ffmpeg_files)} files)')
else:
    print('[spec] ffmpeg が見つかりません。同梱しません')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=_np_binaries + _sp_binaries + _mp_binaries + _ort_binaries + _ffmpeg_files,
    datas=[(_ctk_dir, 'customtkinter')] + _np_datas + _sp_datas + _mp_datas + _ort_datas + _asset_datas,
    hiddenimports=['comtypes.stream'] + _np_hiddenimports + _sp_hiddenimports + _mp_hiddenimports + _ort_hiddenimports + _ai_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchaudio', 'torchvision', 'tensorflow', 'keras'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GolfSwingBroadcast',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['*.pyd', '*.dll'],
    name='GolfSwingBroadcast',
)
