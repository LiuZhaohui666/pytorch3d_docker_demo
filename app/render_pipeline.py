import os
import time
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from box_renderer import BoxRenderer
from pytorch3d.structures import Pointclouds

def load_pointcloud_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if colors.shape[0] == 0:
        colors = np.ones_like(points)

    # ✅ 新增：将点居中并缩放到 -1~1 区间
    center = points.mean(axis=0)
    points -= center
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= scale

    pc = torch.from_numpy(points).float()
    feat = torch.from_numpy(colors).float()
    return pc, feat


def save_image(tensor_img, save_path):
    img = tensor_img[..., :3].cpu().numpy()
    print(f"[DEBUG] Image max: {img.max():.4f}, min: {img.min():.4f}")
    plt.imsave(save_path, img)

def main():
    input_root = "/input_clouds"
    output_root = "/rendered_views"
    os.makedirs(output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = BoxRenderer(device=device, img_size=(256, 256))
    view_names = renderer.view_names

    for subdir in sorted(os.listdir(input_root)):
        ply_path = os.path.join(input_root, subdir, "cloud_denoise.ply")
        if not os.path.exists(ply_path):
            continue

        print(f"\n[INFO] Loading point cloud: {ply_path}")
        pc, feat = load_pointcloud_ply(ply_path)
        print(f"[INFO] Loaded point cloud with {pc.shape[0]} points")

        pc = pc.to(device)
        feat = feat.to(device)
        pointcloud = Pointclouds(points=[pc], features=[feat])

        for i, view in enumerate(view_names):
            print(f"[INFO] Rendering view: {view}")
            img = renderer.render_single_view(i, pointcloud)[0]
            save_path = os.path.join(output_root, f"{subdir}_{view}.png")
            save_image(img, save_path)
            print(f"[INFO] Saved: {save_path}")

if __name__ == "__main__":
    main()
