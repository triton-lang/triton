# print every command
# set -o xtrace

# set path
DOCKERFILE_PATH=scripts/docker/Dockerfile.triton_rocm
# DOCKERFILE_PATH=scripts/docker/Dockerfile.triton_cuda

# get tag
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
DOCKERIMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f 2- -d '.')
echo $DOCKERIMAGE_NAME

# build docker
docker build -f $DOCKERFILE_PATH -t $DOCKERIMAGE_NAME .
