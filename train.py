import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
from time import time
sys.path.append("./emd/")
import emd_module as emd
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = 'weights/weights.pth',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 8192,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
torch.cuda.empty_cache()

class FullModel(nn.Module):
    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters):
        output1, output2, expansion_penalty = self.model(inputs)
        gt = gt[:, :, :3] 
        
        dist, _ = self.EMD(output1, gt, eps, iters)
        emd1 = torch.sqrt(dist).mean(1)
        
        dist, _ = self.EMD(output2, gt, eps, iters)
        emd2 = torch.sqrt(dist).mean(1)    

        return output1, output2, emd1, emd2, expansion_penalty
    

def main():
    opt = parser.parse_args()
    print(opt)
    start_time = time.time()

    now = datetime.datetime.now()
    save_path = now.isoformat()

    opt.manualSeed = random.randint(1, 10000) 
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    best_val_loss = 10

    dataset = ShapeNet(train=True, npoints=opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    dataset_test = ShapeNet(train=False, npoints=opt.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    len_dataset = len(dataset)
    print("Train Set Size: ", len_dataset)

    network = MSN(num_points = opt.num_points, n_primitives = opt.n_primitives)
    network = torch.nn.DataParallel(FullModel(network))
    network.cuda()
    network.module.model.apply(weights_init) #initialization of the weight

    if opt.model != '':
        network.module.model.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    lrate = 0.001 #learning rate
    optimizer = optim.Adam(network.module.model.parameters(), lr = lrate)

    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()

    train_curve = []
    val_curve = []
    labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
    labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
    labels_generated_points = labels_generated_points.contiguous().view(-1)

    print('Starting Process')

    for epoch in range(opt.nepoch):
        # TRAIN MODE
        train_loss.reset()
        network.module.model.train()
        
        # Learning rate schedule
        if epoch == 20:
            optimizer = optim.Adam(network.module.model.parameters(), lr = lrate/10.0)
        if epoch == 40:
            optimizer = optim.Adam(network.module.model.parameters(), lr = lrate/100.0)

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            id, input, gt = data
            input = input.float().cuda()
            gt = gt.float().cuda()
            input = input.transpose(2,1).contiguous()

            _, output2, emd1, emd2, expansion_penalty  = network(input, gt.contiguous(), 0.005, 50)         
            loss_net = emd1.mean() + emd2.mean() + expansion_penalty.mean() * 0.1
            loss_net.backward()
            train_loss.update(emd2.mean().item())
            optimizer.step()
            print('[%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f spend_time: %f' % (epoch+1, i, len_dataset/opt.batchSize, emd1.mean().item(), emd2.mean().item(), expansion_penalty.mean().item(), time.time()-start_time))
            
        train_curve.append(train_loss.avg)
        with torch.no_grad():
            points = output2.cpu().numpy()[0]
            p2 = o3d.geometry.PointCloud()
            p2.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud('test.ply', p2) #f'outputs_test//test{epoch}.ply', p2) 

        # if (epoch+1)%5==0:
        #     # Save model weights
        #     print('Saving net...')
        torch.save(network.module.model.state_dict(), 'end3.pth')#f'weights/5shots_to_completepcd{epoch+1}.pth')
        torch.save(network.module.model.state_dict(), r'Z:\Samani\end3.pth')

if __name__ == '__main__':
    main()
