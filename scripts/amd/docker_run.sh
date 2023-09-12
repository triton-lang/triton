set -o xtrace

DRUN='sudo docker run -it --rm --network=host --user root --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

# DEVICES="--gpus all"
DEVICES="--device=/dev/kfd --device=/dev/dri"

MEMORY="--ipc=host --shm-size 16G"

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR='/root/$(basename $(pwd))'
WORK_DIR="/dockerx/$(basename $(pwd))"

IMAGE_NAME=rocm/pytorch-nightly:latest
# IMAGE_NAME=rocm/pytorch:latest
# IMAGE_NAME=nvcr.io/nvidia/pytorch

CONTAINER_NAME=triton

# start new container
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
CONTAINER_ID=$($DRUN -d -w $WORK_DIR --name $CONTAINER_NAME $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
# docker cp . $CONTAINER_ID:$WORK_DIR
# docker exec $CONTAINER_ID bash -c "bash scripts/amd/run.sh"
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
