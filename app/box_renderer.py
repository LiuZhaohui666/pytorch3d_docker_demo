import torch
from torch import nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    NormWeightedCompositor,
    look_at_view_transform,
)

class PointsRenderer(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds)
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
        )
        return images.permute(0, 2, 3, 1)

class BoxRenderer:
    def __init__(self, device, img_size=(256, 256), radius=0.01, points_per_pixel=5):
        self.device = device
        self.img_size = img_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.view_names = ["top", "front", "back", "left", "right"]
        self._init_cameras()

    def _init_cameras(self):
        elev_azim = {
            "top": (0, 0),
            "front": (90, 0),
            "back": (270, 0),
            "left": (0, 90),
            "right": (0, 270),
        }

        elev = torch.tensor([v[0] for v in elev_azim.values()])
        azim = torch.tensor([v[1] for v in elev_azim.values()])
        up = [(0, 1, 0) if name not in ["left", "right"] else (0, 0, 1) for name in elev_azim]

        self.R, self.T = look_at_view_transform(dist=1.0, elev=elev, azim=azim, up=up)

    def render_single_view(self, view_idx: int, pointcloud):
        camera = FoVOrthographicCameras(
            device=self.device,
            R=self.R[view_idx : view_idx + 1],
            T=self.T[view_idx : view_idx + 1],
            znear=0.01,
        )
        raster_settings = PointsRasterizationSettings(
            image_size=self.img_size,
            radius=self.radius,
            points_per_pixel=self.points_per_pixel,
            bin_size=0,
        )
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        compositor = NormWeightedCompositor()
        renderer = PointsRenderer(rasterizer, compositor)
        return renderer(pointcloud)
