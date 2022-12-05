rm -rf ./scripts/amd/hip_kernel.out
rm -rf ./scripts/amd/temps
mkdir ./scripts/amd/temps
# hipcc -save-temps=./scripts/amd/temps scripts/amd/hip_kernel.cpp -o scripts/amd/hip_kernel.out
hipcc -ffast-math -save-temps=./scripts/amd/temps scripts/amd/hip_kernel.cpp -o scripts/amd/hip_kernel.out
./scripts/amd/hip_kernel.out