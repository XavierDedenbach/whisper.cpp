# Native amd64/arm64 build; Spark's CUDA 13/Blackwell PyTorch comes from NVIDIA.
ARG CUDA_BASE=nvcr.io/nvidia/pytorch:25.09-py3@sha256:f3b4a33fd60d5e0358287bb44d57e91eda45e8f1681fef1bfd6c969371417130
FROM ${CUDA_BASE}
ARG PHONON_REF=cf97fdbedaf5bfd3cc8b88c946e64c0fdc2587b1
RUN apt-get update -qq && apt-get install -y --no-install-recommends git gcc python3-dev \
    && git clone --no-checkout --filter=blob:none https://github.com/fermionresearch/phonon.git /tmp/phonon \
    && git -C /tmp/phonon checkout "${PHONON_REF}" -- docker-phonon2 LICENSE \
    && mkdir -p /opt/phonon && cp /tmp/phonon/docker-phonon2/*.py /opt/phonon/ \
    && cp /tmp/phonon/LICENSE /opt/phonon/LICENSE
# The NVIDIA base owns torch; do not replace it with PyPI's ARM CPU wheel.
RUN PIP_CONSTRAINT= pip install --no-cache-dir "transformers==5.17.0" "numpy<3" soundfile huggingface_hub zstandard \
    && useradd --create-home --uid 1001 phonon \
    && mkdir -p /home/phonon/.cache && chown phonon:phonon /home/phonon/.cache
ENV HOME=/home/phonon PYTHONUNBUFFERED=1 PHONON2_CUDA_PAD_S=5 PHONON2_CUDA_K=16 PHONON2_CUDA_DTYPE=bfloat16
USER phonon
WORKDIR /home/phonon
ENTRYPOINT ["python", "/opt/phonon/entrypoint.py"]
