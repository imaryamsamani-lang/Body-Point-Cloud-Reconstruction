import open3d as o3d
import numpy as np
import glob
import torch
from scipy.spatial import cKDTree

from model import *
from utils import *

num_points = 8192
n_primitives = 16
model = r'\\pnas\ai\Samani\end3.pth' #'weights/5shots_to_halfpcd.pth'
modelc = r'weights/5shots_to_completepcd.pth'

network = MSN(num_points = num_points, n_primitives = n_primitives) 
network.cuda()
network.apply(weights_init)
network.load_state_dict(torch.load(model))
network.eval()


networkc = MSN(num_points = num_points, n_primitives = n_primitives) 
networkc.cuda()
networkc.apply(weights_init)
networkc.load_state_dict(torch.load(modelc))
networkc.eval()

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def NoiseRemover(PointCloud, Radius: float, MinPoints: float):
        PointCloud = np.asarray(PointCloud.points)
        kdtree = cKDTree(PointCloud)
        filtered_points = []
        for point in PointCloud:
            num_neighbors = len(kdtree.query_ball_point(point, Radius))
            if num_neighbors >= MinPoints:
                filtered_points.append(point)
        filtered_points = np.array(filtered_points)
        modified_pcd = o3d.geometry.PointCloud()
        modified_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        return modified_pcd

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def Preprocess(i, dirty_pcd, no, depth_distance_remove=-1):
        dirty_pcd = np.array([point for point in dirty_pcd if (point[2] > depth_distance_remove) & (point[2] < -0.5)])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dirty_pcd)
        
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)

        if i < (no/2):
            modified_pcd = pcd.select_by_index(np.where(np.logical_or(normals[:, 0] > 0.4, normals[:, 2] > 0.9))[0])
        else:
            modified_pcd = pcd
        
        return modified_pcd

files = glob.glob(r'\\pnas\ai\Clean models\models\*') #r'C:\Users\test\Desktop\40test\*')
files.reverse() 

for i in files:

    with torch.no_grad():

        partial = torch.zeros((1, 5000, 3), device='cuda')

        pcd = o3d.geometry.PointCloud()

        p = []

        p.append(o3d.io.read_point_cloud(i + '/1.ply'))
        p.append(o3d.io.read_point_cloud(i + '/2.ply'))
        p.append(o3d.io.read_point_cloud(i + '/3.ply'))
        p.append(o3d.io.read_point_cloud(i + '/4.ply'))
        p.append(o3d.io.read_point_cloud(i + '/5.ply'))
        
        point_clouds = []
        
        for s in range(5):
          p[s], _ = preprocess_point_cloud(Preprocess(s, np.array(p[s].points), 5), voxel_size=0.01)
          p[s] = NoiseRemover(p[s], 0.1, 50)

          points = np.array(p[s].points)
          points[:, [0,1, 2]] = points[:, [2,0, 1]]

          min_z = np.min(points[:, 2])
          points[:, 2] -= min_z
          min_y = np.mean(points[:, 1])
          points[:, 1] -= min_y
          min_x = np.mean(points[:, 0])
          points[:, 0] -= min_x

          p[s].points = o3d.utility.Vector3dVector(points)
          p[s] = resample_pcd(torch.from_numpy(np.array(p[s].points)).float(), 20000)

          point_clouds.append(p[s])

        
        total_points = torch.cat(point_clouds, dim=0)

        pcd.points = o3d.utility.Vector3dVector(total_points)

        partial[0, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), 5000))
        _, output2, _ = network(partial.transpose(2,1).contiguous())

        _, output2c, _ = networkc(partial.transpose(2,1).contiguous())
         
        points = output2.cpu().numpy()[0]
    
        p2 = o3d.geometry.PointCloud()
        p2.points = o3d.utility.Vector3dVector(points)

        pointsc = output2c.cpu().numpy()[0]
    
        p2c = o3d.geometry.PointCloud()
        p2c.points = o3d.utility.Vector3dVector(pointsc)


        p2.paint_uniform_color([0, 0, 0])
        p2c.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([p2, p2c])