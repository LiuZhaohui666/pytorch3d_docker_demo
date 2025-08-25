
# 🧱 PyTorch3D Point Cloud Multi-View Renderer (Dockerized for macOS/Linux)

This project renders `.ply` point clouds from multiple virtual camera angles using [PyTorch3D](https://pytorch3d.org/). It is designed to run inside a Docker container, making it fully portable across macOS, Linux, and Windows (via WSL or Docker Desktop).

---

## ✨ Features

- 🔭 Multi-view rendering (top, front, oblique, etc.)
- 📸 Easily customizable camera configuration (`box_renderer.py`)
- 📁 Batch process any number of input clouds
- 🐳 Docker-based: no GPU or native PyTorch3D setup required on macOS
- ✅ Includes a minimal test script to verify PyTorch3D functionality (`render_pointcloud.py`)

---

## 🧪 Quick Test (Optional)

To **verify that PyTorch3D is correctly installed**, run:

docker run -it --rm \
  -v /Users/zhaohui/Projects/pytorch3d_docker_demo/app:/workspace \
  -v /Users/zhaohui/Projects/FoundationStereo/250805_vis_demo/output:/input_clouds \
  -v /Users/zhaohui/Projects/FoundationStereo/250805_vis_demo/rendered_views:/rendered_views \
  -w /workspace \
  rvt2-renderer bash

python render_pipeline.py


This will generate a simple render using synthetic point data and save to `output/output_render.png`.

---

## 🚀 How to Run Full Multi-View Pipeline

### 1. Build Docker Image

```bash
docker build -t rvt2-renderer .
```

### 2. Run Rendering

```bash
docker run -it --rm \
  -v /absolute/path/to/input:/workspace/input_clouds \
  -v /absolute/path/to/output:/workspace/rendered_views \
  -v /absolute/path/to/code:/workspace \
  -w /workspace \
  rvt2-renderer \
  python render_pipeline.py
```

---

## 📥 Input Format

Each point cloud should be placed in its own folder, named with an index:

```
input_clouds/
├── 0/
│   └── cloud_denoise.ply
├── 1/
│   └── cloud_denoise.ply
...
```

---

## 📸 Output Example

Each view generates a separate image:

```
rendered_views/
├── 0_top.png
├── 0_front.png
├── 0_oblique_135.png
├── 1_top.png
...
```

You can define or expand views in `box_renderer.py`:

```python
self.view_configs = {
    "top": (90, 0, (0, 0, -1), 1.0),
    "front_high": (30, 180, (0, 1, 0), 1.0),
    ...
}
```

---

## 🧱 Project Layout

```
.
├── app/
│   ├── render_pipeline.py    # Main script to render all views
│   ├── box_renderer.py       # Handles camera setup and rendering
│   └── render_pointcloud.py  # (Optional) test script for PyTorch3D
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📄 License

MIT License.

---

## 🙋‍♂️ Need Help?

If you're unsure whether PyTorch3D is running correctly, run `render_pointcloud.py` to verify your setup before debugging the full pipeline.
