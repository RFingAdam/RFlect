# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['anthropic', 'PIL._tkinter_finder']
hiddenimports += collect_submodules('scipy')


a = Analysis(
    ['run_rflect.py'],
    pathex=[],
    binaries=[],
    datas=[('settings.json', '.'), ('assets/smith_logo.png', 'assets')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'huggingface_hub', 'tokenizers', 'safetensors',
        'tensorflow', 'keras',
        'IPython', 'jupyter', 'notebook', 'ipykernel', 'ipywidgets',
        'cv2', 'sklearn', 'scikit-learn',
        'sympy', 'bokeh', 'plotly', 'dash',
        'pytest', 'mypy', 'black', 'flake8',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RFlect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['smith_logo.ico'],
)
