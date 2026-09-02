# -*- mode: python ; coding: utf-8 -*-
import glob, importlib, os
from PyInstaller.utils.hooks import collect_all
_ctk_dir = os.path.dirname(importlib.import_module('customtkinter').__file__)

# Collect all submodules, binaries and data for numpy/scipy 2.x compatibility
_np_datas, _np_binaries, _np_hiddenimports = collect_all('numpy')
_sp_datas, _sp_binaries, _sp_hiddenimports = collect_all('scipy')

# 軌道自動検出 (YOLO) とスロー化 (RIFE) は torch/ultralytics が要る。
# 除外すると autotrace は動かず、スロー化も ffmpeg フォールバックに
# 落ちて実尺33倍まで遅くなる。同梱するとサイズは数GBになる。
_ul_datas, _ul_binaries, _ul_hidden = collect_all('ultralytics')

# ffmpeg 同梱 (EXEフォルダだけで動作するように)
_ffmpeg_dir = r'C:\ffmpeg\bin'
_ffmpeg_files = [(f, '.') for f in
                 glob.glob(os.path.join(_ffmpeg_dir, 'ffmpeg.exe')) +
                 glob.glob(os.path.join(_ffmpeg_dir, '*.dll'))
                 if os.path.exists(f)]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=_np_binaries + _sp_binaries + _ffmpeg_files + _ul_binaries,
    datas=[(_ctk_dir, 'customtkinter')] + _np_datas + _sp_datas + _ul_datas,
    hiddenimports=(['comtypes.stream', 'autotrace', 'slowmo',
                    'rife_slowmo', 'rife_trt']
                   + _np_hiddenimports + _sp_hiddenimports + _ul_hidden),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torchaudio', 'tensorflow', 'keras', 'matplotlib'],
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
