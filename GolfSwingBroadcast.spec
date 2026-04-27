# -*- mode: python ; coding: utf-8 -*-
import importlib, os
_ctk_dir = os.path.dirname(importlib.import_module('customtkinter').__file__)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[(_ctk_dir, 'customtkinter')],
    hiddenimports=['comtypes.stream'],
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
