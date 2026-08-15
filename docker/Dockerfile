# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS builder

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libprotobuf-dev \
    protobuf-compiler \
    libopencv-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装ONNX Runtime
WORKDIR /tmp
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
RUN tar -xzf onnxruntime-linux-x64-1.14.1.tgz
RUN mv onnxruntime-linux-x64-1.14.1 /opt/onnxruntime

# 构建项目
WORKDIR /workspace
COPY . .

# 创建构建目录
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRuntime_DIR=/opt/onnxruntime \
        -DCMAKE_INSTALL_PREFIX=/opt/rangenet && \
    make -j$(nproc) && \
    make install

# 运行时镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libopencv-core4.2 \
    libopencv-imgproc4.2 \
    && rm -rf /var/lib/apt/lists/*

# 复制ONNX Runtime
COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /opt/rangenet /opt/rangenet

# 设置环境变量
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/rangenet/bin:$PATH

# 创建工作目录
WORKDIR /workspace
COPY models/ /workspace/models/

# 默认命令
CMD ["rangenet_inference", "--help"]