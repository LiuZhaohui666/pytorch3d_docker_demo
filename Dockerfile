FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install numpy matplotlib opencv-python

RUN git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && \
    pip install .

WORKDIR /app

CMD ["bash"]
