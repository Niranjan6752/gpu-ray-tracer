# Runs every experiment and sweep for the GPU BVH ray tracer and collects the results.
# Run from the src folder in the Developer PowerShell (the one where nvcc works):
#     powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
# Optional: .\run_experiments.ps1 -Part blocks   (parts: all, blocks, layout, sweeps, heat, brute)
# Output:  results.csv (one RESULT line per run), builds.csv (registers / stack per build), the images in ..\images\output
param([string]$Part = "all")

$ErrorActionPreference = "Continue"
$src = "raytracer_gpu_exp.cu"
$baseFlags = @("-O2", "-arch=sm_75", "-DSCENE=2", "-DNO_BALLS")

function Build([string]$name, [string[]]$defs) {
    if (Test-Path "$name.exe") { Remove-Item "$name.exe" }
    $out = (& nvcc @baseFlags @defs "-Xptxas" "-v" -o "$name.exe" $src 2>&1 | Out-String)
    if (-not (Test-Path "$name.exe")) { Write-Host "BUILD FAILED: $name"; Write-Host $out; return $false }
    $regs = ""; $stack = ""
    if ($out -match "Used (\d+) registers") { $regs = $Matches[1] }
    if ($out -match "(\d+) bytes stack frame") { $stack = $Matches[1] }
    "$name,$($defs -join ' '),registers=$regs,stack_frame_bytes=$stack" | Add-Content builds.csv
    Write-Host "built $name  (registers $regs, stack frame $stack bytes)"
    return $true
}

function Run([string]$name, [string[]]$runArgs) {
    $lines = & ".\$name.exe" @runArgs 2>&1
    foreach ($l in $lines) {
        if ($l -like "RESULT,*") { "$name,$l" | Add-Content results.csv; Write-Host $l }
        elseif ($l -like "HEAT,*" -or $l -like "kernel:*" -or $l -like "Tree depth*") { Write-Host $l }
    }
}

New-Item -ItemType Directory -Force ..\images\output | Out-Null
if (-not (Test-Path results.csv)) { "name,RESULT_fields..." | Set-Content results.csv }
if (-not (Test-Path builds.csv))  { "name,flags,registers,stack_frame" | Set-Content builds.csv }

# ---- 1. block shape sweep (experiment 1) -----------------------------------
if ($Part -in "all", "blocks") {
    foreach ($b in @(@(8,4), @(8,8), @(16,8), @(16,16), @(32,4), @(32,8), @(32,1))) {
        $n = "blk_$($b[0])x$($b[1])"
        if (Build $n @("-DBLOCK_X=$($b[0])", "-DBLOCK_Y=$($b[1])")) { Run $n @() }
    }
}

# ---- 2. data layout and stack size (experiments 2 and 3) -------------------
if ($Part -in "all", "layout") {
    if (Build "lay0_stack32" @("-DSTACK_SIZE=32"))                  { Run "lay0_stack32" @() }
    if (Build "lay1_stack64" @("-DLAYOUT=1"))                       { Run "lay1_stack64" @() }
    if (Build "lay1_stack32" @("-DLAYOUT=1", "-DSTACK_SIZE=32"))    { Run "lay1_stack32" @() }
}

# ---- 3. sweeps: resolution, bounces, scene size (baseline build, no recompiles) ----
if ($Part -in "all", "sweeps") {
    if (Build "base" @()) {
        foreach ($r in @(@(400,300), @(800,600), @(1600,1200), @(1920,1080))) { Run "base" @("$($r[0])", "$($r[1])") }
        foreach ($b in 0, 1, 3, 5, 10) { Run "base" @("-1", "-1", "$b") }
        foreach ($c in 1, 2, 4, 8, 16) { Run "base" @("-1", "-1", "-1", "$c") }
    }
}

# ---- 4. divergence heat map -------------------------------------------------
if ($Part -in "all", "heat") {
    if (Build "base" @()) { Run "base" @("-1", "-1", "-1", "1", "1", "1") }
}

# ---- 5. GPU brute force at the small size (the 2 s Windows watchdog limits larger images) ----
if ($Part -in "all", "brute") {
    $brute = @("-O2", "-arch=sm_75", "-DSCENE=1", "-DBRUTE")
    if (Test-Path "brute.exe") { Remove-Item "brute.exe" }
    & nvcc @brute -o brute.exe $src 2>&1 | Out-Null
    if (Test-Path "brute.exe") { Run "brute" @("200", "150") } else { Write-Host "BUILD FAILED: brute" }
    $bvh = @("-O2", "-arch=sm_75", "-DSCENE=1")
    if (Test-Path "bvh_small.exe") { Remove-Item "bvh_small.exe" }
    & nvcc @bvh -o bvh_small.exe $src 2>&1 | Out-Null
    if (Test-Path "bvh_small.exe") { Run "bvh_small" @("200", "150") } else { Write-Host "BUILD FAILED: bvh_small" }
}

Write-Host ""
Write-Host "Done. Send me results.csv and builds.csv (or paste them)."
