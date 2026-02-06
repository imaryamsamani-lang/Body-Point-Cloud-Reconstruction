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
            self.data = os.listdir('data/train')[2000:2500]
            self.tot = 'train/'
        else:
            self.data = os.listdir('data/test')
            self.tot = 'test/'

        self.npoints = npoints
        self.k = 2.2
        self.len = len(self.data)
 
    def read_pcd(self, filename):
        pcd = o3d.io.read_point_cloud(filename)
        return torch.from_numpy(np.array(pcd.points)).float(), pcd

    def make_data(self, pcd_main, k, j, angle, noise):

        p = []
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(np.array([point for point in np.array(pcd_main.points) if (point[2] < k/5 + k/j)]))
        p.append(pcd1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.array([point for point in np.array(pcd_main.points) if (point[2] > k/5 - k/j) & (point[2] < 2*k/5 + k/j)]))
        p.append(pcd2)
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(np.array([point for point in np.array(pcd_main.points) if (point[2] > 2*k/5 - k/j) & (point[2] < 3*k/5 + k/j)]))
        p.append(pcd3)
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(np.array([point for point in np.array(pcd_main.points) if (point[2] > 3*k/5 - k/j) & (point[2] < 4*k/5 + k/j)]))
        p.append(pcd4)
        pcd5 = o3d.geometry.PointCloud()
        pcd5.points = o3d.utility.Vector3dVector(np.array([point for point in np.array(pcd_main.points) if (point[2] > 4 * k/5 - k/j)]) if np.any([point for point in np.array(pcd_main.points) if (point[2] > 4 * k/5 - k/j)]) else np.empty((0, 3)))
        p.append(pcd5)


        # Convert angle to radians
        angle_radians = np.deg2rad(angle)

        # Rotation matrix around the z-axis
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
        
        point_clouds = []
        
        if len(np.array(p[4].points))>0:
          counter = 5
          
        else:
          counter = 4

        for s in range(counter):
            points = np.array(p[s].points)
            min_z = np.min(points[:, 2])
            points[:, 2] -= min_z
            min_y = np.mean(points[:, 1])
            points[:, 1] -= min_y
            min_x = np.mean(points[:, 0])
            points[:, 0] -= min_x
            # WritingDataToPly(f'{s}.ply', np.array(points))

            combined_points = self.add_noise_to_points(points, noise)
            p[s].points = o3d.utility.Vector3dVector(combined_points)
            # WritingDataToPly(f'{s}N.ply', np.array(p[s].points))

            p[s] = p[s].rotate(rotation_matrix, center=(0, 0, 0))
            # WritingDataToPly(f'{s}T.ply', np.array(p[s].points))

            p[s] = self.resample_pcd(torch.from_numpy(np.array(p[s].points)).float(), 5000)
            # WritingDataToPly(f'{s}R.ply', np.array(p[s]))

            
            point_clouds.append(p[s])

        total_points = torch.cat(point_clouds, dim=0)

        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(total_points)

        return torch.from_numpy(np.array(c.points)).float()

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

        complete, main = self.read_pcd('data/partial/' + self.tot + model_id + '.ply')
        partial = self.make_data(main, self.k, random.uniform(5, 8), random.uniform(-10, 10), 0.005)
        #complete, main = self.read_pcd(r'Data\datasets\main_models/' + self.tot + model_id + '.ply')

        return model_id, self.resample_pcd(partial, 5000), self.resample_pcd(complete, self.npoints)

    def __len__(self):

        return self.len