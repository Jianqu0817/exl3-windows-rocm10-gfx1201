$ErrorActionPreference = 'Stop'
$portRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$runtimeSite = 'D:\LLM_local\tools\exl3-gfx1201\python\Lib\site-packages'
$env:PYTHONNOUSERSITE = '1'
if (!(Test-Path "$runtimeSite\torch\__init__.py")) { throw 'Existing ROCm 10.1 runtime is required' }
if (!(Test-Path "$portRoot\build\.venv\Scripts\python.exe")) {
    uv venv --python 3.12 "$portRoot\build\.venv"
    if ($LASTEXITCODE) { throw 'uv venv failed' }
}
uv pip install --python "$portRoot\build\.venv\Scripts\python.exe" --prerelease allow `
    --index https://rocm.prereleases.amd.com/whl-multi-arch/ --extra-index-url https://pypi.org/simple `
    --index-strategy unsafe-best-match 'torch[device-gfx1201]==2.13.0+rocm10.0.0rc4' `
    'rocm[libraries,devel]==10.0.0rc4' ninja setuptools wheel numpy pybind11
if ($LASTEXITCODE) { throw 'Build dependencies failed' }
& "$portRoot\build\.venv\Scripts\python.exe" -m rocm_sdk path --root
if ($LASTEXITCODE) { throw 'SDK expansion failed' }
