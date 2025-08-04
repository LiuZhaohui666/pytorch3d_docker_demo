FROM python:3.10-slim

# 安装系统依赖（OpenGL、CMake 等）
RUN apt-get update && apt-get install -y \
    git build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch（CPU 版本）
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他 Python 依赖，包括 open3d
RUN pip install numpy matplotlib opencv-python open3d

# 安装 PyTorch3D（从 GitHub 源码安装）
RUN git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && \
    pip install .

WORKDIR /app

CMD ["bash"]