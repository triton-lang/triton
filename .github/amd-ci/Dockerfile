# syntax=docker/dockerfile:1

# Build using the `build.py` script. It records complete argument sets for the
# images used in CI.

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG ROCM_VERSION
ARG ROCM_RELEASE_TYPE
ARG ROCM_REPO_DIRECTORY
ARG PYTORCH_VERSION
ARG PYTORCH_INDEX_URL
ARG HIP_PYTHON_VERSION
ARG PYTORCH_GPU_TARGETS

LABEL org.opencontainers.image.title="Triton AMD CI image"
LABEL org.opencontainers.image.description="ROCm, PyTorch, and OS dependencies for Triton AMD CI"
LABEL org.opencontainers.image.source="https://github.com/triton-lang/triton"
LABEL org.opencontainers.image.version="rocm${ROCM_VERSION}-pytorch${PYTORCH_VERSION}"
LABEL org.triton-lang.image.rocm.version="${ROCM_VERSION}"
LABEL org.triton-lang.image.rocm.repo-directory="${ROCM_REPO_DIRECTORY}"
LABEL org.triton-lang.image.pytorch.version="${PYTORCH_VERSION}"
LABEL org.triton-lang.image.gpu-targets="${PYTORCH_GPU_TARGETS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY --chmod=0755 .github/amd-ci/install-system-packages.sh /usr/local/bin/
RUN install-system-packages.sh

COPY --chmod=0755 .github/amd-ci/install-rocm.sh /usr/local/bin/
RUN install-rocm.sh

COPY --chmod=0755 .github/amd-ci/install-python-packages.sh /usr/local/bin/
RUN install-python-packages.sh

CMD ["/bin/bash"]
