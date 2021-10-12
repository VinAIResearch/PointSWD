import numpy as np
import open3d
import torch
from open3d.open3d.geometry import voxel_down_sample
from tqdm import tqdm


class Preprocessor:
    def __init__(self, trained_ae, device="cuda"):
        super().__init__()
        self.autoencoder = trained_ae
        self.device = device

    def extract_points_and_features(self, pcd, voxel_size, radius, num_points, batch_size=256, color=False):
        points, patches = self._extract_uniform_patches(pcd, voxel_size, radius, num_points, color)
        features = self._compute_features(patches, batch_size)
        return points, features

    def _extract_uniform_patches(self, pcd, voxel_size, radius, num_points, color=False):
        kdtree = open3d.geometry.KDTreeFlann(pcd)
        downsampled_points = voxel_down_sample(pcd, voxel_size)
        points = np.asarray(downsampled_points.points)
        patches = []
        for i in range(points.shape[0]):
            k, index, _ = kdtree.search_hybrid_vector_3d(points[i], radius, num_points)
            if k < num_points:
                index = np.random.choice(index, num_points, replace=True)
            xyz = np.asarray(pcd.points)[index]
            xyz = (xyz - points[i]) / radius  # normalize to local coordinates
            if color:
                rgb = np.asarray(pcd.colors)[index]
                patch = np.concatenate([xyz, rgb], axis=1)
            else:
                patch = xyz
            patches.append(patch)
        patches = np.stack(patches, axis=0)
        return points, patches

    def _compute_features(self, patches, batch_size):
        batches = torch.tensor(patches, dtype=torch.float32)
        batches = torch.split(batches, batch_size)
        features = []
        self.autoencoder.eval()
        with torch.no_grad():
            for _, x in tqdm(enumerate(batches)):
                x = x.to(self.device)
                z = self.autoencoder.encode(x)
                features.append(z.cpu().numpy())
        return np.concatenate(features, axis=0)
