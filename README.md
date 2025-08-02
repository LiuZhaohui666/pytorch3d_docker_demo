# PyTorch3D in Docker (macOS Compatible Demo)

This project demonstrates how to run [PyTorch3D](https://pytorch3d.org/) on macOS using Docker, since PyTorch3D is not natively supported on macOS.

## 🚀 Why this project?

- PyTorch3D is officially unsupported on macOS
- This Docker-based setup runs PyTorch3D (CPU-only) inside a Linux container on macOS
- No need for GPU or Conda — pure Docker + Python 3.10

## 📦 Features

- Fully containerized PyTorch3D setup
- Generates and renders a random 3D point cloud
- Saves rendered image to disk
- Works on **macOS**, **Linux**, **Windows** (via Docker)

## 🛠 How to Use

### 1. Build the Docker image

```bash
docker build -t pytorch3d_cpu .
```

### 2. Run the demo script

```bash
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  pytorch3d_cpu \
  python3 app/render_pointcloud.py
```

### 3. Output

- The image will be saved to `output/output_render.png`
- You can open it using Preview or any image viewer on macOS

## 🗂 Project Structure

```
.
├── app/
│   └── render_pointcloud.py
├── output/
│   └── output_render.png
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

## 📌 Notes

- This demo uses **CPU-only** PyTorch and PyTorch3D.
- It does **not** require GPU or CUDA.
- Rendering performance is sufficient for small demos.

## 📜 License

MIT License.
