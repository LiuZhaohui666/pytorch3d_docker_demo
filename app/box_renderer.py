# box_renderer.py
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
)
from pytorch3d.renderer import look_at_view_transform
import torch
import torch.nn as nn


class PointsRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, with_depth=False, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)

        if with_depth:
            depth = fragments.zbuf[..., 0]
            _, h, w = depth.shape
            depth_0 = depth == -1
            depth_sum = torch.sum(depth, (1, 2)) + torch.sum(depth_0, (1, 2))
            depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (1, 2)))
            depth -= depth_mean.unsqueeze(-1).unsqueeze(-1)
            depth[depth_0] = -1

        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        images = images.permute(0, 2, 3, 1)

        if with_depth:
            images = torch.cat((images, depth.unsqueeze(-1)), dim=-1)

        return images


class BoxRenderer:
    def __init__(
        self,
        device,
        img_size=(256, 256),
        radius=0.012,
        points_per_pixel=5,
        compositor="norm",
        with_depth=False,
    ):
        self.device = device
        self.img_size = img_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.compositor = compositor
        self.with_depth = with_depth
        self._init_renderer()

    def _init_renderer(self):
        raster_settings = PointsRasterizationSettings(
            image_size=self.img_size,
            radius=self.radius,
            points_per_pixel=self.points_per_pixel,
        )

        if self.compositor == "norm":
            compositor = NormWeightedCompositor()
        else:
            compositor = AlphaCompositor()

        elev_azim = [
            (0, 0),    # top
            (90, 0),   # front
            (270, 0),  # back
            (0, 90),   # left
            (0, 270),  # right
        ]

        self._fix_cam = []
        self._fix_ren = []

        for elev, azim in elev_azim:
            R, T = look_at_view_transform(dist=1.0, elev=elev, azim=azim)
            cam = FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01)
            self._fix_cam.append(cam)

            rasterizer = PointsRasterizer(cameras=cam, raster_settings=raster_settings)
            renderer = PointsRendererWithDepth(rasterizer, compositor)
            self._fix_ren.append(renderer)

        self.num_fix_cam = len(self._fix_ren)

    def render_single_view(self, index: int, pointcloud: Pointclouds):
        assert 0 <= index < self.num_fix_cam
        return self._fix_ren[index](pointcloud, with_depth=self.with_depth)
