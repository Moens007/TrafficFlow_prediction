import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data  import DataLoader
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from traffic_dataset import LoadData
from random_seed import set_seed
from chebnet import ChebNet
from visualize_data import get_flow
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear1 = nn.Linear(in_c,hid_c)
        self.linear2 = nn.Linear(hid_c,out_c)
        self.act = nn.ReLU()


    def forward(self, data, device):
        graph_data = data['graph'].to(device)[0]  #[N,N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data['flow_x'].to(device)
        B, N, H, D = flow_x.size()
        flow_x = flow_x.view([B, N, -1])  # [B,N,H*D]  H=6,D=1
        out_put1 = self.linear1(flow_x)   # [B,N,hid_C]

        out_put1 = self.act(torch.matmul(graph_data, out_put1))   #[N,N] [B,N,hid_C]   批量的mm

        out_put2 = self.linear2(out_put1)
        out_put2 = self.act(torch.matmul(graph_data, out_put2))    #[B,N,OUT_C]->[B,N,1,OUT_C]

        return out_put2.unsqueeze(2)



    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  #[N,N]  A~

        degree_matrix = torch.sum(graph_data,dim = -1,keepdim=False) #[N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float('inf') ] = 0.  #[N]
        degree_matrix = torch.diag(degree_matrix)  #[N,N] 对角化

        return torch.mm(graph_data, degree_matrix)  #D^(-1) * A = \hat A




class Baseline(nn.Module):
    def __init__(self,in_c,out_c):
        super(Baseline,self).__init__()
        self.layer = nn.Linear(in_c,out_c)

    def forward(self,data,device):
        flow_x = data['flow_x'].to(device)  #[B,N,H,D]
        B, N, H, D = flow_x.size()
        flow_x = flow_x.view(B,N,-1)  #[B,N,H*D]
        y = self.layer(flow_x)  #[B,N,out_C]，out_c = D
        return y.unsqueeze(2) #[B,N,1,D=out_C]




def main():
    set_seed(2020)

    train_Data =  LoadData(data_path=['../PeMS_04\PeMS04.csv', '../PeMS_04\PeMS04.npz'], num_nodes=307,
                      divide_days=[45, 14], time_interval=5, history_length=6, train_mode='train')
    train_loader = DataLoader(train_Data, batch_size=64, shuffle=True,)

    test_Data = LoadData(data_path=['../PeMS_04\PeMS04.csv', '../PeMS_04\PeMS04.npz'], num_nodes=307,
                          divide_days=[45, 14], time_interval=5, history_length=6, train_mode='test')
    test_loader = DataLoader(test_Data, batch_size=64, shuffle=True)

    #load.model
    # my_net = GCN(in_c = 6 , hid_c = 6, out_c = 1)
    # my_net = Baseline(in_c=6,out_c=1)
    my_net = ChebNet(in_C=6, hid_C=6,out_c=1,K = 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params= my_net.parameters(),)


    #train model
    my_net.train()
    Epoch = 100
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for i,data in enumerate(train_loader):  #['graph':[B,N,N], 'flow_x':[B,N,H,D], 'flow_y':[B,N,1,D]]
            my_net.zero_grad()
            predict_value = my_net(data, device).to(torch.device('cpu'))  #[B,N,1,D]


            loss = criterion(predict_value, data['flow_y'])
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        end_time = time.time()
        print('Epoch:{:04d},loss:{:02.4f},time:{:02.2f}'.format(epoch,epoch_loss/i,(end_time-start_time)/60))
    state = {'net':my_net.state_dict(),'optim':optimizer.state_dict()}
    torch.save(state,'../model/model_param.pth')

    # Test_Model
    # result visualization
    my_net.eval()
    with torch.no_grad():#关闭梯度
        total_loss = 0.0
        for i,data in enumerate(test_loader):
            predict_value = my_net(data, device).to(torch.device('cpu'))  # [B,N,1,D]
            loss = criterion(predict_value, data['flow_y'])
            total_loss +=loss.item()
        print('Test_loss:{:02.4f}'.format(total_loss/i))


        # traffic_data  = get_flow('../PeMS_04/PeMS04.npz')
        # plt.plot(traffic_data[1, :24 * 12, 0])
        # history_num = 10
        # predict_data = torch.from_numpy(traffic_data[1,:history_num,0])
        # predict_value = traffic_data[1,:history_num,0]
        # print(predict_value)
        # #预测图像
        # for i in range(history_num,24*12):
        #     predict_now = my_net(predict_data, device).to(torch.device('cpu')).item
        #     predict_value.append(predict_now)
        #     predict_data = torch.from_numpy(predict_value)
        #
        # plt.show()


if __name__ == '__main__':
    main()