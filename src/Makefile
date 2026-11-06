# Build and run the CPU and GPU ray tracers. Run from the src folder.
#
#   make help        list the commands below
#   make             build everything
#   make spheres     step 1: spheres + reflections, CPU vs first CUDA port, image compare
#   make run         step 2: bunny with BVH, CPU vs GPU, image compare
#   make brute       step 3: GPU brute force vs GPU BVH (200x150)
#   make cpubrute    step 3b: CPU brute force vs CPU BVH (200x150, slow: about 2 minutes)
#   make opt         step 4: GPU optimisations one by one (1920x1080, median of 9 launches)
#   make sweep       resolution, bounce and scene-size sweeps with the best GPU version
#   make heat        heat maps of BVH nodes / triangles tested per pixel
#   make demo        steps 1, 2, 3 and 4 in one go
#   make clean       remove the built programs (do this after changing a setting)
#
# Settings (e.g. make run SCENE=1 WIDTH=1920 HEIGHT=1080):
#   SCENE   1 matte, 2 default (5 bounces), 3 mirror (10 bounces)
#   WIDTH, HEIGHT   image size for spheres and run
#   BALLS   0 = bunny only (as in the report), 1 = keep the three balls
#   ARCH    GPU architecture, sm_75 for Turing (GTX 16xx / RTX 20xx / T4)

SCENE  ?= 2
WIDTH  ?= 800
HEIGHT ?= 600
BALLS  ?= 0
ARCH   ?= sm_75

CXX  ?= g++
NVCC ?= nvcc
PY   ?= python3

DEFS = -DSCENE=$(SCENE) -DWIDTH=$(WIDTH) -DHEIGHT=$(HEIGHT)
ifeq ($(BALLS),0)
DEFS += -DNO_BALLS
endif

CXXFLAGS  = -O2 $(DEFS)
NVCCFLAGS = -O2 -arch=$(ARCH) $(DEFS)
SMALL_CPU = -O2 -DSCENE=1 -DNO_BALLS -DWIDTH=200 -DHEIGHT=150
SMALL_GPU = -O2 -arch=$(ARCH) -DSCENE=1 -DNO_BALLS -DWIDTH=200 -DHEIGHT=150

OUT   = ../images/output
MKOUT = -@mkdir -p $(OUT)

# On Windows (MSYS2 g++) a static build avoids a runtime DLL clash
ifeq ($(OS),Windows_NT)
CXXFLAGS  += -static
SMALL_CPU += -static
EXE   = .exe
PY    = python
endif

CPU_PROGS = cpu_fresnel cpu_mesh cpu_bvh cpu_bvh_small cpu_brute_small
GPU_PROGS = gpu_fresnel gpu_bvh gpu_bvh_small gpu_brute_small \
            gpu_exp_base gpu_exp_8x8 gpu_exp_lay1 gpu_exp_lay1_8x4 gpu_exp
PROGS = $(addsuffix $(EXE),$(CPU_PROGS) $(GPU_PROGS))

# 1920x1080, scene 2, median of 9 launches (args: W H bounces copies runs heat)
EXP_ARGS = 1920 1080 -1 1 9

.PHONY: all help spheres run brute cpubrute opt sweep heat demo clean outdir

all: $(PROGS)

help:
	@echo make spheres / run / brute / cpubrute / opt / sweep / heat / demo / clean
	@echo settings: SCENE=1..3 WIDTH=.. HEIGHT=.. BALLS=0/1 ARCH=sm_75

# "make gpu_bvh" also works on Windows, where the file is gpu_bvh.exe
ifdef EXE
.PHONY: $(CPU_PROGS) $(GPU_PROGS)
$(CPU_PROGS) $(GPU_PROGS): %: %$(EXE)
endif

# ---------------- CPU programs ----------------
cpu_fresnel$(EXE): raytracer_cpu_fresnel_reflection.cpp config.h
	$(CXX) $(CXXFLAGS) -o $@ $<

cpu_mesh$(EXE): raytracer_cpu_mesh.cpp config.h
	$(CXX) $(CXXFLAGS) -o $@ $<

cpu_bvh$(EXE): raytracer_cpu_bvh.cpp config.h
	$(CXX) $(CXXFLAGS) -o $@ $<

cpu_bvh_small$(EXE): raytracer_cpu_bvh.cpp config.h
	$(CXX) $(SMALL_CPU) -o $@ $<

cpu_brute_small$(EXE): raytracer_cpu_bvh.cpp config.h
	$(CXX) $(SMALL_CPU) -DBRUTE -o $@ $<

# ---------------- GPU programs ----------------
gpu_fresnel$(EXE): raytracer_gpu_fresnel_reflections.cu config.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu_bvh$(EXE): raytracer_gpu_bvh.cu config.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu_bvh_small$(EXE): raytracer_gpu_bvh.cu config.h
	$(NVCC) $(SMALL_GPU) -o $@ $<

gpu_brute_small$(EXE): raytracer_gpu_bvh.cu config.h
	$(NVCC) $(SMALL_GPU) -DBRUTE -o $@ $<

# experiment build, one program per optimisation step
gpu_exp_base$(EXE): raytracer_gpu_exp.cu config.h
	$(NVCC) $(NVCCFLAGS) -DLAYOUT=0 -DBLOCK_X=16 -DBLOCK_Y=16 -o $@ $<

gpu_exp_8x8$(EXE): raytracer_gpu_exp.cu config.h
	$(NVCC) $(NVCCFLAGS) -DLAYOUT=0 -DBLOCK_X=8 -DBLOCK_Y=8 -o $@ $<

gpu_exp_lay1$(EXE): raytracer_gpu_exp.cu config.h
	$(NVCC) $(NVCCFLAGS) -DLAYOUT=1 -DBLOCK_X=16 -DBLOCK_Y=16 -o $@ $<

gpu_exp_lay1_8x4$(EXE): raytracer_gpu_exp.cu config.h
	$(NVCC) $(NVCCFLAGS) -DLAYOUT=1 -DBLOCK_X=8 -DBLOCK_Y=4 -o $@ $<

# best version: float4 layout, 8x8 blocks
gpu_exp$(EXE): raytracer_gpu_exp.cu config.h
	$(NVCC) $(NVCCFLAGS) -DLAYOUT=1 -DBLOCK_X=8 -DBLOCK_Y=8 -o $@ $<

# ---------------- commands ----------------
outdir:
	$(MKOUT)

spheres: cpu_fresnel$(EXE) gpu_fresnel$(EXE) outdir
	@echo ==== spheres + reflections: CPU vs first GPU port, $(WIDTH)x$(HEIGHT), scene $(SCENE)
	./cpu_fresnel$(EXE)
	./gpu_fresnel$(EXE)
	$(PY) compare.py $(OUT)/cpu.ppm $(OUT)/gpu.ppm

run: cpu_bvh$(EXE) gpu_bvh$(EXE) outdir
	@echo ==== bunny + BVH: CPU vs GPU, $(WIDTH)x$(HEIGHT), scene $(SCENE)
	./cpu_bvh$(EXE)
	./gpu_bvh$(EXE)
	$(PY) compare.py $(OUT)/cpu_bvh.ppm $(OUT)/gpu_bvh.ppm

brute: gpu_brute_small$(EXE) gpu_bvh_small$(EXE) outdir
	@echo ==== GPU brute force vs GPU BVH, 200x150, scene 1
	./gpu_brute_small$(EXE)
	./gpu_bvh_small$(EXE)

cpubrute: cpu_brute_small$(EXE) cpu_bvh_small$(EXE) outdir
	@echo ==== CPU brute force vs CPU BVH, 200x150, scene 1 - brute force takes about 2 minutes
	./cpu_brute_small$(EXE)
	./cpu_bvh_small$(EXE)

opt: gpu_exp_base$(EXE) gpu_exp_8x8$(EXE) gpu_exp_lay1$(EXE) gpu_exp_lay1_8x4$(EXE) gpu_exp$(EXE) outdir
	@echo ==== GPU optimisations, 1920x1080, scene $(SCENE), median of 9 launches
	@echo ---- 1. first version: layout 0, 16x16 blocks
	./gpu_exp_base$(EXE) $(EXP_ARGS)
	@echo ---- 2. 8x8 blocks
	./gpu_exp_8x8$(EXE) $(EXP_ARGS)
	@echo ---- 3. float4 layout, 16x16 blocks
	./gpu_exp_lay1$(EXE) $(EXP_ARGS)
	@echo ---- 4. float4 layout, 8x4 blocks: occupancy capped at 50 percent
	./gpu_exp_lay1_8x4$(EXE) $(EXP_ARGS)
	@echo ---- 5. float4 layout, 8x8 blocks: best
	./gpu_exp$(EXE) $(EXP_ARGS)

sweep: gpu_exp$(EXE) outdir
	@echo ==== resolution sweep
	./gpu_exp$(EXE) 400 300 -1 1 9
	./gpu_exp$(EXE) 800 600 -1 1 9
	./gpu_exp$(EXE) 1920 1080 -1 1 9
	./gpu_exp$(EXE) 3840 2160 -1 1 9
	@echo ==== bounce sweep, 1920x1080
	./gpu_exp$(EXE) 1920 1080 0 1 9
	./gpu_exp$(EXE) 1920 1080 1 1 9
	./gpu_exp$(EXE) 1920 1080 5 1 9
	./gpu_exp$(EXE) 1920 1080 10 1 9
	@echo ==== scene size: 1, 4 and 16 bunnies, 1920x1080
	./gpu_exp$(EXE) 1920 1080 -1 1 9
	./gpu_exp$(EXE) 1920 1080 -1 4 9
	./gpu_exp$(EXE) 1920 1080 -1 16 9

heat: gpu_exp$(EXE) outdir
	@echo ==== heat maps: writes $(OUT)/heat_nodes.ppm and heat_tris.ppm
	./gpu_exp$(EXE) 1920 1080 -1 1 1 1

demo: spheres run brute opt

clean:
	rm -f $(PROGS)
