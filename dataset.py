import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random

def WritingDataToPly( FilePath , PointCloud):
    flipped_mesh = o3d.geometry.TriangleMesh()
    flipped_mesh.vertices = o3d.utility.Vector3dVector(PointCloud)
    o3d.io.write_triangle_mesh(FilePath, flipped_mesh)  

class ShapeNet(data.Dataset): 
    def __init__(self, train = True, npoints = 8192):
        if train:
            self.data = os.listdir('data/train')
            self.tot = 'train/'
        else:
            self.data = os.listdir('data/test')
            self.tot = 'test/'

        self.npoints = npoints
        self.k = 2.2
        self.len = len(self.data)
 
    def read_pcd(self, filename):
        pcd = o3d.io.read_point_cloud(filename)
        return torch.from_numpy(np.array(pcd.points)).float()

    def resample_pcd(self, pcd, n):
        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
        return pcd[idx[:n]]
    
    def add_noise_to_points(self,points, noise, noise_fraction=0.005):
        # Choose random indices from the points array
        num_points = points.shape[0]
        num_noise_points = int(num_points * noise_fraction)
        random_indices = np.random.choice(num_points, num_noise_points, replace=False)

        noise_points = points[random_indices]
        noise_points += np.random.normal(0, noise, noise_points.shape)
        combined_points = np.vstack((points, noise_points))

        return combined_points

    def __getitem__(self, index):
        model_id = self.data[index].split('.')[0]

        complete = self.read_pcd('data/train/main/' + self.tot + model_id + '.ply')
        partial = self.read_pcd('data/train/partial/' + self.tot + model_id + '.ply')

        return model_id, self.resample_pcd(partial, 5000), self.resample_pcd(complete, self.npoints)

    def __len__(self):
        
        return self.len
