"""
Render a random point cloud using PyTorch3D and save it as an image.
"""

import torch
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    FoVPerspectiveCameras,
    look_at_view_transform,
)

# Generate a random 3D point cloud with RGB colors
points = torch.rand((1, 1000, 3))
colors = torch.rand((1, 1000, 3))
point_cloud = Pointclouds(points=points, features=colors)

# Set top-down camera view
R, T = look_at_view_transform(dist=2.0, elev=90.0, azim=0.0)
camera = FoVPerspectiveCameras(R=R, T=T)

# Rasterization settings
raster_settings = PointsRasterizationSettings(
    image_size=512,
    radius=0.01,
    points_per_pixel=10
)

# Build renderer
renderer = PointsRenderer(
    rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
    compositor=AlphaCompositor()
)

# Render the point cloud to an image
image = renderer(point_cloud)[0, ..., :3].detach().numpy()

# Save output image
plt.imshow(image)
plt.title("Rendered Point Cloud (Top View)")
plt.axis("off")
plt.savefig("output/output_render.png")
print("✅ Saved: output/output_render.png")
