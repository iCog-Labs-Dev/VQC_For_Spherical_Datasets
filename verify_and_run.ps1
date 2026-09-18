# verify_and_run.ps1 -- run from the repository root:  .\verify_and_run.ps1
# Verifies the branch applied, then runs the training-free stages.
# Nothing here trains a model; expect a few minutes total.

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

# Prefer the repo's virtualenv if it exists, else whatever python is on PATH.
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "`n=== interpreter ===" -ForegroundColor Cyan
& $py --version
& $py -c "import pennylane, numpy, scipy, sklearn, matplotlib; print('pennylane', pennylane.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| sklearn', sklearn.__version__, '| mpl', matplotlib.__version__)"

Write-Host "`n=== 1. branch state (expect 33 new commits, newest first) ===" -ForegroundColor Cyan
git log --oneline -5
git status --short --branch

Write-Host "`n=== 2. test suite (expect 78 passed) ===" -ForegroundColor Cyan
& $py -m pytest tests -q

Write-Host "`n=== 3. dataset gate + analytic core (no training) ===" -ForegroundColor Cyan
& $py -m vqc_spherical.RunAll --phase 0

Write-Host "`n=== 4. kernel spectrum (no training) ===" -ForegroundColor Cyan
& $py -m vqc_spherical.ExperimentKernelSpectrum

Write-Host "`n=== what to check above ===" -ForegroundColor Yellow
Write-Host "  tests            : 78 passed"
Write-Host "  dataset gate     : 'Every generator satisfies the test for its declared role.'"
Write-Host "  quadrupole ceil  : ~0.784   (majority baseline ~0.575)"
Write-Host "  affine residuals : ~1e-15 at every depth and both qubit counts"
Write-Host "  Gram rank        : 4, eigenvalues/n ~ 0.500, 0.176, 0.162, 0.161"
Write-Host "  spin ladder rank : 4, 9, 16, 25 for n = 1..4"
Write-Host ""
Write-Host "Next, the decisive one (~30 min, 10 seeds, three rotation axes):" -ForegroundColor Yellow
Write-Host "  $py -m vqc_spherical.ExperimentFaithfulness"
Write-Host ""
