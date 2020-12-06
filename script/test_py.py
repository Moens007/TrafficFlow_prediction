import csv
import torch
import numpy as np
from traffic_dataset import get_adjacent_matrix,get_flow_data

# A = get_adjacent_matrix(distance_file = '../PeMS_04/PeMS04.csv',num_nodes = 307,id_file = None,graph_type = 'distance')
# print(A)
# B = get_flow_data('../PeMS_04/PeMS04.npz')
# print(B.shape)

# A = np.array([[1,3],[1,3],[2,4]])
# A = torch.tensor(A).unsqueeze(1)
# print(A.shape)
A = torch.Tensor([[3,3],[2,2]])
B = torch.sum(A,dim = -1,keepdim=False)
B = B.pow(-1)

A = torch.eye(3)
degree_matrix = torch.diag(B)  #[N,N] 对角化

print(A)