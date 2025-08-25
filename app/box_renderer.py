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
        #self.view_names = ["top", "front", "back", "left", "right"]
        self.view_configs = {
            # 顶视
            "top": (90, 0, (0, 0, -1), 1.0),
            "top_far": (90, 0, (0, 0, -1), 1.5),
            "top_close": (90, 0, (0, 0, -1), 0.5),

            # 前视 + 高低角度
            "front": (0, 180, (0, 1, 0), 1.0),
            "front_high": (30, 180, (0, 1, 0), 1.0),
            "front_low": (-30, 180, (0, 1, 0), 1.0),

            # 左右
            "left": (0, 90, (0, 1, 0), 1.0),
            "right": (0, -90, (0, 1, 0), 1.0),

            # 斜视（45° / 135°）
            "oblique_45": (45, 45, (0, 1, 0), 1.2),
            "oblique_135": (45, 135, (0, 1, 0), 1.2),

            # 🔥 新增：更多角度 🔥
            # 环绕四周 (elev=15°)
            "view15_a0": (15, 0, (0, 1, 0), 1.0),
            "view15_a90": (15, 90, (0, 1, 0), 1.0),
            "view15_a180": (15, 180, (0, 1, 0), 1.0),
            "view15_a270": (15, 270, (0, 1, 0), 1.0),

            # 环绕四周 (elev=45°)
            "view45_a0": (45, 0, (0, 1, 0), 1.0),
            "view45_a90": (45, 90, (0, 1, 0), 1.0),
            "view45_a180": (45, 180, (0, 1, 0), 1.0),
            "view45_a270": (45, 270, (0, 1, 0), 1.0),

            # 环绕四周 (elev=60°)
            "view60_a0": (60, 0, (0, 1, 0), 1.0),
            "view60_a90": (60, 90, (0, 1, 0), 1.0),
            "view60_a180": (60, 180, (0, 1, 0), 1.0),
            "view60_a270": (60, 270, (0, 1, 0), 1.0),
        }

        self._init_cameras()

    #def _init_cameras(self):
     #   elev_azim = {
      #      "top": (0, 0),
       #     "front": (90, 0),
        #    "back": (270, 0),
         #   "left": (0, 90),
          #  "right": (0, 270),
        #}

        #elev = torch.tensor([v[0] for v in elev_azim.values()])
        #azim = torch.tensor([v[1] for v in elev_azim.values()])
        #up = [(0, 1, 0) if name not in ["left", "right"] else (0, 0, 1) for name in elev_azim]

        #self.R, self.T = look_at_view_transform(dist=1.0, elev=elev, azim=azim, up=up)
    def _init_cameras(self):
        names = []
        R_all, T_all = [], []

        for name, (elev, azim, up, dist) in self.view_configs.items():
            R, T = look_at_view_transform(
                dist=dist,
                elev=torch.tensor([elev]),
                azim=torch.tensor([azim]),
                up=torch.tensor([up], dtype=torch.float)
            )
            names.append(name)
            R_all.append(R)
            T_all.append(T)

        self.view_names = names
        self.R = torch.cat(R_all, dim=0)
        self.T = torch.cat(T_all, dim=0)

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
