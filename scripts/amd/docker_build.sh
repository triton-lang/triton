# print every command
set -o xtrace

# set path
# DOCKERFILE_PATH=scripts/docker/Dockerfile.triton_rocm
# DOCKERFILE_PATH=scripts/docker/Dockerfile.triton_cuda
# DOCKERFILE_PATH=triton_rocm_all_archs.Dockerfile
DOCKERFILE_PATH=triton_rocm_20-52.Dockerfile

# get tag
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
DOCKERIMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f -1 -d '.')
echo $DOCKERIMAGE_NAME

# build docker
docker build --build-arg CACHEBUST=$(date +%s) -f $DOCKERFILE_PATH -t $DOCKERIMAGE_NAME .
