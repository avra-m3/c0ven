# Two targets (ADR-0025). `engine`: the image the engine is built and tested in, ADR-0002's pinned set, as a
# non-root user. `serve`: the same image with the engine built inside, the image the engine runs from.
# Build and run: README.md, "The image".

FROM ubuntu:26.04@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 AS engine

ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu packages come from the archive as it was on this date, so versions from -updates stay installable.
ARG UBUNTU_SNAPSHOT=20261013T000000Z

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates=20260223 \
 && apt-get update --snapshot "$UBUNTU_SNAPSHOT" \
 && apt-get install -y --no-install-recommends \
      g++-15=15.2.0-16ubuntu1 gcc-15=15.2.0-16ubuntu1 \
      cmake=4.2.3-2ubuntu2 ninja-build=1.13.2-1 \
      python3.14=3.14.4-1ubuntu0.2 python3=3.14.3-0ubuntu2 \
      gdb=17.1-2ubuntu1 strace=6.19+ds-0ubuntu5 \
      git curl make pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 100 \
 && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-15 100 \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-15 100

# The GPU stack and uv, from their release pages, checked against ADR-0002's checksums.
WORKDIR /tmp/debs
RUN set -e; \
    cr=https://github.com/intel/compute-runtime/releases/download/26.40.40196.8; \
    igc=https://github.com/intel/intel-graphics-compiler/releases/download/v2.42.6; \
    ze=https://github.com/oneapi-src/level-zero/releases/download/v1.32.0; \
    curl -fsSLO "$cr/libze-intel-gpu1_26.40.40196.8-0_amd64.deb"; \
    curl -fsSLO "$cr/libigdgmm12_22.10.2_amd64.deb"; \
    curl -fsSLO "$cr/intel-ocloc_26.40.40196.8-0_amd64.deb"; \
    curl -fsSLO "$igc/intel-igc-core-2_2.42.6+23070_amd64.deb"; \
    curl -fsSLO "$igc/intel-igc-opencl-2_2.42.6+23070_amd64.deb"; \
    curl -fsSLO "$ze/libze1_1.32.0+u24.04_amd64.deb"; \
    curl -fsSLO "$ze/libze-dev_1.32.0+u24.04_amd64.deb"; \
    curl -fsSLO https://github.com/astral-sh/uv/releases/download/0.12.22/uv-x86_64-unknown-linux-gnu.tar.gz; \
    printf '%s\n' \
      'd9c86b2bb195baea6936c2708846cd1fceeadc39bc5e167f728858a5ca6fb86e  libze-intel-gpu1_26.40.40196.8-0_amd64.deb' \
      '053fceadb84a1d2f9a30a0d1bc9b16424b3f1a90cc49900fcd80aadcf25c8a4d  libigdgmm12_22.10.2_amd64.deb' \
      'b2868d6d7958cc5ca28fa9b48a7d39961760ee39ef07673d5b58894dd53d6b4e  intel-ocloc_26.40.40196.8-0_amd64.deb' \
      'e20539632f2d99875d8f8b225d5da58b31f98baa54bb47c6328aa98ec473ab9d  intel-igc-core-2_2.42.6+23070_amd64.deb' \
      '7f5593cf828503ffb8450967768bde9209e22753bc576296e846b5a751011fa3  intel-igc-opencl-2_2.42.6+23070_amd64.deb' \
      '3c846af24f84a89150f6a4c6adcb4ea4ebef74dc119fe44f4e269bfaa72c7ba6  libze1_1.32.0+u24.04_amd64.deb' \
      '4b783ed5fb937a55a7a0f3a8bc66af252f362e82476ebc0304da36173c9f2eb8  libze-dev_1.32.0+u24.04_amd64.deb' \
      'b9980552309f09c15172b8be828555e375097f16deb459795ce7bfd200380f0b  uv-x86_64-unknown-linux-gnu.tar.gz' \
      | sha256sum -c -; \
    tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz -C /usr/local/bin --strip-components=1; \
    apt-get update --snapshot "$UBUNTU_SNAPSHOT"; \
    apt-get install -y --no-install-recommends ./*.deb; \
    rm -rf /var/lib/apt/lists/* /tmp/debs

# ggml at the commit ADR-0012 pins, for builds without network access.
RUN git init -q /opt/llama.cpp \
 && git -C /opt/llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp \
      3cf03257f219afbe7334045ff7c6a06ac68c627d \
 && git -C /opt/llama.cpp checkout -q FETCH_HEAD \
 && git config --system --add safe.directory /opt/llama.cpp

# ADR-0002's environment row; the engine sets the debug keys itself. uv uses the pinned Python, never its own.
ENV ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \
    ZES_ENABLE_SYSMAN=1 \
    UV_PYTHON=/usr/bin/python3.14 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    FLASH_MODEL_DIR=/models \
    FLASH_REF_DIR=/scratch/probes

# The base image's user, uid 1000. The render nodes' group is added at run time (--group-add).
RUN mkdir -p /work /home/ubuntu/.cache && chown ubuntu:ubuntu /work /home/ubuntu/.cache
USER ubuntu
WORKDIR /work
CMD ["bash"]

LABEL project=c0ven

# The checkout, the build and the venv, owned by root and read-only to the user, so the root filesystem can be
# mounted read-only. The build's paths are compiled into the core (ENGINE_KERNEL_DIR, ENGINE_QWEN4EXP_ROUTING_PROFILE), so
# they never move. ocloc compiles the kernels without a card (ADR-0028).
FROM engine AS serve
USER root
WORKDIR /opt/c0ven
COPY CMakeLists.txt VERSION pyproject.toml uv.lock .python-version protocol.py ./
COPY cmake cmake
COPY include include
COPY shared shared
COPY kernels kernels
COPY core core
COPY tools tools
COPY engine engine
COPY frontend frontend
# The committed tensor tables and MTP manifest the memory plan reads at run time (engine/models/qwen4exp/draft.py,
# tensors.py); without them a DFlash 2 drafter's plan fails.
COPY docs/model docs/model
RUN cmake -S /opt/c0ven -B /opt/c0ven/build -G Ninja \
 && ninja -C /opt/c0ven/build \
 && UV_NO_CACHE=1 uv sync --locked --no-dev

ENV ENGINE_LIBRARY=/opt/c0ven/build/core/libengine.so \
    PATH=/opt/c0ven/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1

# The arguments of `docker run` are the front end's flags (frontend/__main__.py). The model comes in at /models.
USER ubuntu
ENTRYPOINT ["python", "-m", "frontend"]
