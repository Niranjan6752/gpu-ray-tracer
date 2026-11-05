# run every version in order, same laptop, log to progression_log.txt
# usage (from src): powershell -ExecutionPolicy Bypass -File .\run_progression.ps1

$o = "..\images\output"
mkdir -Force $o | Out-Null
Start-Transcript progression_log.txt -Force | Out-Null

function run($exe, $n, $a = @()) {
    for ($i = 1; $i -le $n; $i++) { & ".\$exe" @a 2>&1 | Select-String "ms|Loaded|BVH|RESULT" }
}

# 1. spheres + reflections, cpu vs gpu, scene 2, 800x600
echo "`n== 1. spheres cpu vs gpu"
g++ -O2 -static -DSCENE=2 -DWIDTH=800 -DHEIGHT=600 -o p1_cpu.exe raytracer_cpu_fresnel_reflection.cpp
nvcc -O2 -arch=sm_75 -DSCENE=2 -DWIDTH=800 -DHEIGHT=600 -o p1_gpu.exe raytracer_gpu_fresnel_reflections.cu
run p1_cpu.exe 3; run p1_gpu.exe 3
python compare.py $o/cpu.ppm $o/gpu.ppm

# 2. bunny, brute force vs bvh, scene 1, 200x150 (brute force cpu is slow)
echo "`n== 2. bunny brute vs bvh, 200x150"
$f = "-DSCENE=1", "-DWIDTH=200", "-DHEIGHT=150"
g++ -O2 -static @f -o p2_mesh.exe raytracer_cpu_mesh.cpp
g++ -O2 -static @f -DNO_BALLS -DBRUTE -o p2_cpu_brute.exe raytracer_cpu_bvh.cpp
g++ -O2 -static @f -DNO_BALLS -o p2_cpu_bvh.exe raytracer_cpu_bvh.cpp
nvcc -O2 -arch=sm_75 @f -DNO_BALLS -DBRUTE -o p2_gpu_brute.exe raytracer_gpu_bvh.cu
nvcc -O2 -arch=sm_75 @f -DNO_BALLS -o p2_gpu_bvh.exe raytracer_gpu_bvh.cu
echo "-- cpu_mesh (old brute force, has balls)"; run p2_mesh.exe 1
echo "-- cpu brute"; run p2_cpu_brute.exe 1; cp $o/cpu_bvh.ppm $o/p2_cpu_brute.ppm
echo "-- cpu bvh";   run p2_cpu_bvh.exe 3;   cp $o/cpu_bvh.ppm $o/p2_cpu_bvh.ppm
echo "-- gpu brute"; run p2_gpu_brute.exe 3; cp $o/gpu_bvh.ppm $o/p2_gpu_brute.ppm
echo "-- gpu bvh";   run p2_gpu_bvh.exe 3;   cp $o/gpu_bvh.ppm $o/p2_gpu_bvh.ppm
python compare.py $o/p2_cpu_brute.ppm $o/p2_cpu_bvh.ppm
python compare.py $o/p2_gpu_brute.ppm $o/p2_gpu_bvh.ppm
python compare.py $o/p2_cpu_bvh.ppm $o/p2_gpu_bvh.ppm

# 3. bunny + reflections, cpu bvh vs gpu bvh, scene 2, 800x600
echo "`n== 3. bunny cpu vs gpu, 800x600"
$f = "-DSCENE=2", "-DNO_BALLS", "-DWIDTH=800", "-DHEIGHT=600"
g++ -O2 -static @f -o p3_cpu.exe raytracer_cpu_bvh.cpp
nvcc -O2 -arch=sm_75 @f -o p3_gpu.exe raytracer_gpu_bvh.cu
run p3_cpu.exe 3; run p3_gpu.exe 3
python compare.py $o/cpu_bvh.ppm $o/gpu_bvh.ppm

# 4. gpu optimisations one at a time, scene 2, 1920x1080, median of 9
echo "`n== 4. gpu optimisations, 1920x1080"
$v = @{ "p4_base_16x16" = "-DLAYOUT=0 -DBLOCK_X=16 -DBLOCK_Y=16";
        "p4_lay0_8x8"   = "-DLAYOUT=0 -DBLOCK_X=8 -DBLOCK_Y=8";
        "p4_lay1_16x16" = "-DLAYOUT=1 -DBLOCK_X=16 -DBLOCK_Y=16";
        "p4_lay1_8x4"   = "-DLAYOUT=1 -DBLOCK_X=8 -DBLOCK_Y=4";
        "p4_lay1_8x8"   = "-DLAYOUT=1 -DBLOCK_X=8 -DBLOCK_Y=8" }
foreach ($k in $v.Keys | Sort-Object) {
    echo "-- $k"
    nvcc -O2 -arch=sm_75 -DSCENE=2 -DNO_BALLS $v[$k].Split(" ") -o "$k.exe" raytracer_gpu_exp.cu
    run "$k.exe" 1 @("1920", "1080", "-1", "1", "9")
}

Stop-Transcript | Out-Null
