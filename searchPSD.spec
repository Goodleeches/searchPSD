# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['autoImageMerge.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[ 
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='autoImageMerge',
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
    onefile=True,  # 단일 파일로 빌드하기 위해 onefile=True 추가
)
