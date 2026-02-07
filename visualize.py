import open3d as o3d
import numpy as np
import glob
import torch

from model import *
from utils import *

num_points = 8192
n_primitives = 16
model =  'weights/halfpcd_to_completepcd.pth'

network = MSN(num_points = num_points, n_primitives = n_primitives) 
network.cuda()
network.apply(weights_init)
network.load_state_dict(torch.load(model))
network.eval()

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

files = glob.glob(r'D:\half_models\*')
#files.reverse()

for i in files:

    with torch.no_grad():

        partial = torch.zeros((1, 5000, 3), device='cuda')
        
        main = o3d.io.read_point_cloud('data/test/main/' + i.split('\\')[-1].split('.')[0] + '.ply')  
        pcd = o3d.io.read_point_cloud('data/test/partial/' + i.split('\\')[-1].split('.')[0] + '.ply')

        partial[0, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), 5000))
        _, output2, _ = network(partial.transpose(2,1).contiguous())
         
        points = output2.cpu().numpy()[0]
    
        p2 = o3d.geometry.PointCloud()
        p2.points = o3d.utility.Vector3dVector(points)

        p2.paint_uniform_color([1, 0, 0])
        main.paint_uniform_color([0, 0, 0])


        o3d.visualization.draw_geometries([main, p2])
