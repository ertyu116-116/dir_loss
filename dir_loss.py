import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D



class ClusteringLossSingle(nn.Module):
    # key operation such as "sort", "scatter_add" works more smoothly with single batch
    # plus memory requirement single batch
    def __init__(self, delta_v=0.5, delta_d=1.5, w_var=1, w_dist=1, w_reg=0.001):
        super(ClusteringLossSingle, self).__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.w_var = w_var
        self.w_dist = w_dist
        self.w_reg = w_reg
        pass

    @staticmethod
    def unsorted_segmented_sum(p, idx, n):
        N, d = p.shape
        rep_idx = idx.unsqueeze(-1).repeat([1, d])
        return torch.zeros([n, d]).to(p.device).scatter_add(0, rep_idx, p)

    def forward(self, predictions, labels):
        N, d = predictions.shape
        unique_labels, unique_id, counts = torch.unique(labels, return_inverse=True, return_counts=True, sorted=False)
        num_instances = torch.numel(unique_labels)

        segmented_sum = self.unsorted_segmented_sum(predictions, unique_id, num_instances)
        mu = segmented_sum / counts.view([num_instances, -1]).type(torch.float)
        mu_expand = mu[unique_id, :]  # [N, d]

        ### calculate l var
        distance = torch.norm(predictions - mu_expand, p=2, dim=-1, keepdim=True)  # [N]
        distance = distance - self.delta_v
        distance = torch.clamp_min(distance, 0) ** 2

        l_var = self.unsorted_segmented_sum(distance, unique_id, num_instances)
        l_var = l_var / counts.view([num_instances, -1]).type(torch.float)
        l_var = torch.sum(l_var) / num_instances

        ### calculate l dist
        mu_interleaved_rep = mu.repeat([num_instances, 1])
        mu_band_rep = mu.repeat([1, num_instances]).view([num_instances ** 2, d])

        mu_diff = mu_band_rep - mu_interleaved_rep
        mu_dist = torch.norm(mu_diff, p=2, dim=-1)
        mu_dist = torch.clamp_min(2. * self.delta_d - mu_dist, 0) ** 2
        # mask out all diagonal elements
        mask = torch.diag(torch.ones([num_instances], dtype=torch.float)).flip(1).view(-1).to(predictions.device)

        l_dist = torch.sum(mu_dist * mask) / torch.sum(mask)

        l_reg = torch.mean(torch.norm(mu, p=2, dim=-1))
        #print('l_var: {}, l_dist: {}, l_reg: {}'.format(l_var, l_dist, l_reg))
        param_scale = 1
        l_var = self.w_var * l_var
        l_dist = self.w_dist * l_dist
        l_reg = self.w_reg * l_reg

        loss = param_scale * (l_var + l_dist + l_reg)
        #make ground_truth_vector
        distance = torch.norm(predictions-mu_expand,p=2,dim=1,keepdim=True)
        gro_tru_vec = (predictions - mu_expand)/ distance # toward center of pointcloud

        return loss,gro_tru_vec,mu_expand,num_instances


class ClusteringLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, w_var=1, w_dist=1, w_reg=0.001):
        super(ClusteringLoss, self).__init__()
        self.single_loss = ClusteringLossSingle(delta_v, delta_d, w_var, w_dist, w_reg)

    def forward(self, predictions, labels):
        B = predictions.shape[0]
        losses = 0
        for i in range(B):
            losses += (self.single_loss(predictions[i], labels[i]))
        return losses / B



if __name__ == "__main__":
    B, N1, N2 = 4, 1000, 4000
    P1 = 10 *torch.randn([B, N1, 3]) + 50 * torch.ones([1, 1, 3]) #50 ->1000
    P2 = 10 *torch.randn([B, N2, 3]) - 50 * torch.ones([1, 1, 3]) #-50 > 4000
    P = torch.cat([P1, P2], dim=1)  # [B, 5000, 3]

    L1 = torch.ones([B, N1])
    L2 = 100 * torch.ones([B, N2])
    L = torch.cat([L1, L2], dim=1)

    loss = ClusteringLossSingle()
    loss, gro_tru_vec,mu_expand,num_instances = loss(P[0],L[0])

    #random sample
    random_list1 = list(range(0,N1,1))
    random_list2 = list(range(N1,N1+N2,1))

    S1 = 30
    S2 = 70

    random_sample1_1 = random.sample(random_list1,S1) #50->1000
    random_sample1_2 = random.sample(random_list2,S2)
    random_sample2_1 = random.sample(random_list1,S1) #-50->4000
    random_sample2_2 = random.sample(random_list2,S2)

    #make pred_vec & gro_tru_vec
    pred_vec = torch.cat([gro_tru_vec[random_sample1_1],gro_tru_vec[random_sample1_2]],dim=0)
    gro_tru_vec = torch.cat([gro_tru_vec[random_sample2_1],gro_tru_vec[random_sample2_2]],dim=0)

    #make random sample for Ground_truth
    random_G1 = P[0,random_sample2_1]
    random_G2 = P[0,random_sample2_2]
    random_G = torch.cat([random_G1,random_G2],dim=0)

    #make random sample for P
    random_P1 = P[0,random_sample1_1]
    random_P2 = P[0,random_sample1_2]
    random_P = torch.cat([random_P1,random_P2],dim=0)

    #make random sample for mu
    random_mu1 = mu_expand[random_sample2_1]
    random_mu2 = mu_expand[random_sample2_2]
    random_mu = torch.cat([random_mu1,random_mu2],dim=0)

    idx1 = 0
    idx2 = 0
    #calculate dir_loss
    for i in range(0,S1,1):#instance1
        dir_vec1 = torch.matmul(pred_vec[i],gro_tru_vec[i])
        idx1 = idx1 + dir_vec1

    idx1 =  idx1 / S1

    for i in range(S1,S1+S2,1):#instance2
        dir_vec2 = torch.matmul(pred_vec[i],gro_tru_vec[i])
        idx2 = idx2 + dir_vec2

    idx2 = idx2 / S2

    dir_loss = -(idx1+idx2)/num_instances

    print("dir_loss1 is {:2f}".format(dir_loss))

    dir_vec3 = torch.matmul(pred_vec,gro_tru_vec.T)

    diff = torch.diag(dir_vec3) #difference between ground truth and prediction

    instance1= diff[0:S1]
    instance2= diff[S1:S1+S2]
    instance1_sum = torch.sum(instance1)/S1
    instance2_sum = torch.sum(instance2)/S2
    dir_loss2 = -(instance1_sum + instance2_sum) / num_instances
    print("dir_loss2 is {:2f}".format(dir_loss2))


    fig = plt.figure()
    ax = Axes3D(fig)
    cpu_P = P[0].detach().cpu().numpy()
    cpu_L = L[0].detach().cpu().numpy()
    ax.scatter(cpu_P[:, 0], cpu_P[:, 1], cpu_P[:, 2], c=cpu_L)
    plt.show()

    fig2 = plt.figure()
    ax1 = Axes3D(fig2)
    #label for dir_P
    dir_L1 = torch.ones([B, 30])
    dir_L2 = 100 * torch.ones([B, 70])
    dir_L = torch.cat([dir_L1, dir_L2], dim=1)
    #label for mu
    mu_L1 = 2 * torch.ones([B,30])
    mu_L2 = 3 * torch.ones([B,70])
    mu_L = torch.cat([mu_L1,mu_L2],dim=1)
    #visualize zi,zc
    #gpu -> cpu
    dir_G = random_G.detach().cpu().numpy()
    dir_P = random_P.detach().cpu().numpy()
    dir_cpu_L = dir_L[0].detach().cpu().numpy()#directional label
    dir_mu_L = mu_L[0].detach().cpu().numpy()
    random_mu = random_mu.detach().cpu().numpy()


    #scatter
    ax1.scatter(dir_G[0:S1,0],dir_G[0:S1,1],dir_G[0:S1,2],c='purple',label='zi(instance1)')
    ax1.scatter(dir_G[S1:S1+S2, 0], dir_G[S1:S1+S2, 1], dir_G[S1:S1+S2, 2], c='hotpink', label='zi(instance2)')
    ax1.scatter(random_mu[:,0],random_mu[:,1],random_mu[:,2],c='red',label='zc')
    ax1.legend(loc='best')
    ax2 = fig2.gca(projection='3d')
    ax2.quiver(dir_G[0:S1,0],dir_G[0:S1,1],dir_G[0:S1,2],
               random_mu[0:S1,0]-dir_G[0:S1,0],random_mu[0:S1,1]-dir_G[0:S1,1],random_mu[0:S1,2]-dir_G[0:S1,2],
               length=5,arrow_length_ratio=0.5,normalize = True ,colors='darkgreen',label='ground_truth_vecor(instance1)'
              )
    ax2.quiver(dir_G[0:S1, 0], dir_G[0:S1, 1], dir_G[0:S1, 2],
               dir_P[0:S1, 0]-dir_G[0:S1, 0],dir_P[0:S1, 1] - dir_G[0:S1, 1],dir_P[0:S1, 2] - dir_G[0:S1, 2],
               length=5, arrow_length_ratio=0.5, normalize=True, colors='lightgreen',label='prediction vector(instance1)'
               )
    ax2.quiver(dir_G[S1:S1+S2,0],dir_G[S1:S1+S2,1],dir_G[S1:S1+S2,2],
               dir_P[S1:S1+S2,0]-dir_G[S1:S1+S2,0],dir_P[S1:S1+S2,1]-dir_G[S1:S1+S2,1],dir_P[S1:S1+S2,2]-dir_G[S1:S1+S2,2],
               length=5,arrow_length_ratio=0.5,normalize = True ,colors='darkorange',label='ground_truth_vecor(instance2)'
              )
    ax2.quiver(dir_G[S1:S1+S2, 0], dir_G[S1:S1+S2, 1], dir_G[S1:S1+S2, 2],
               random_mu[S1:S1+S2, 0] - dir_P[S1:S1+S2, 0], random_mu[S1:S1+S2, 1] - dir_P[S1:S1+S2, 1],random_mu[S1:S1+S2, 2] - dir_P[S1:S1+S2, 2],
               length=5, arrow_length_ratio=0.5, normalize=True, colors='tan',label='prediction vector(instance2)'
               )
    ax2.legend(loc='best')

    plt.show()





