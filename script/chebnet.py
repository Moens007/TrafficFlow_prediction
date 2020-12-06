import torch
import torch.nn as nn
import torch.nn.init as init


class ChebConv(nn.Module):
    '''The ChebNet convolution operation'''
    def __init__(self,in_c, out_c, K, bias = True,normalize = True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K+1,1,in_c,out_c))  #[K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,1,out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.K = K+1

    def forward(self,inputs,graph):
        '''
        :param inputs:   [B,N,C]
        :param graph:   [N,N]
        :return: convolution results[B,N,D]
        '''
        L = ChebConv.get_laplacian(graph,self.normalize)
        mul_L = self.cheb_ploynomail(L).unsqueeze(1) #[k,1,n,n]

        result = torch.matmul(mul_L,inputs)  #[K,B,N,C]
        result = torch.matmul(result,self.weight)  #[K,B,N,D]
        result = torch.sum(result, dim=0) + self.bias

        return result

    def cheb_ploynomail(self,laplacian):
        '''
        Compute the chebyshev  Polynomail, according to the graph laplacian
        :return:the mutil order chebyshev laplacian,[K,N,N]
        '''
        N = laplacian.size(0)  #[N,N]
        mutil_order_laplacian = torch.zeros([self.K,N,N],device=laplacian.device,dtype=torch.float) #[K,B,N,C]
        mutil_order_laplacian[0] = torch.eye(N,device=laplacian.device,dtype=torch.float)

        if self.K ==1:
            return mutil_order_laplacian
        else:
            mutil_order_laplacian[1] = laplacian
            if self.K == 2:
                return mutil_order_laplacian
            else:
                for k in range(2,self.K):
                    mutil_order_laplacian[k] = 2*torch.mm(laplacian, mutil_order_laplacian[k-1]) - mutil_order_laplacian[k-2]

        return mutil_order_laplacian



    @staticmethod
    def get_laplacian(graph, normalize):
        '''
        return the laplacian of the graph
        :param graph:[N,N],the graph structure without self loop
        :param normalize:whether to used the normalized laplacian
        :return: graph laplacian
        '''
        if normalize:
            D = torch.diag(torch.sum(graph, dim=1) ** (-1/2))
            L = torch.eye(graph.size(0),device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D,graph),D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

class ChebNet(nn.Module):
    def __init__(self,in_C , hid_C, out_c, K):
        '''
        :param in_C:
        :param hid_C:
        :param out_c:
        :param K:  切比雪夫展开数
        '''
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c = in_C, out_c = hid_C, K = K)
        self.conv2 = ChebConv(in_c= hid_C, out_c = out_c, K = K)
        self.act = nn.ReLU()

    def forward(self,data,device):
        graph_data = data['graph'].to(device)[0]  #[N,N]
        flow_x = data['flow_x'].to(device)    #[B,N,H,D]
        B, N  = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B,N,-1)  #[B,N,HXD]
        out_1 = self.act(self.conv1(flow_x,graph_data))
        out_2 = self.act(self.conv2(out_1,graph_data))

        return out_2.unsqueeze(2)