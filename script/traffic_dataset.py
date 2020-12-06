import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type='connect') -> np.array:
    '''

    :param distance_file: str ,path of csv file to save the distance between nodes
    :param num_nodes:int , number of nodes in the graph
    :param id_file:str , path of txt file to save the order of the nodes
    :param graph_type:str, ['connet','distance']
    :return:
        np.array[N,N]
    '''
    A = np.zeros([num_nodes, num_nodes])  # 构造NXN节点数量
    if id_file:
        '''处理存在在矩阵中的真实点'''
        with open(id_file, 'r') as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split('\n'))}

            with open(distance_file, 'r') as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    # 构造邻接矩阵
                    if graph_type == 'connect':
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == 'distance':
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError('graph type is not correct(connect or distance)')
        return A

    reader = pd.read_csv(distance_file).values
    for item in reader:
        if len(item) != 3:
            continue
        i, j, distance = int(item[0]), int(item[1]), float(item[2])

        # 构造邻接矩阵
        if graph_type == 'connect':
            A[i, j] = 1.
            A[j, i] = 1.
        elif graph_type == 'distance':
            A[i, j] = 1. / distance
            A[j, i] = 1. / distance
        else:
            raise ValueError('graph type is not correct(connect or distance)')
    return A


def get_flow_data(flow_file: str) -> np.array:
    '''
    :param flow_file:flow_file:str,path of .npz file to save the traffic flow data
    :return:
        np.array(N,T,D)
    '''
    data = np.load(flow_file)
    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]
    return flow_data


class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        '''
        :param data_path:list ,['graph file name', 'flow data file name'],path to save the data file names;
        :param num_nodes:int, numbers of nodes;
        :param divide_days:list,[days of train data, days of test data],list to divide the original data;
        :param time_interval:int, time interval between two traffic data records(mins);
        :param history_length:int,length of history data to be used;
        :param train_mode:list,['train','test']
        '''

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 59 - 14 = 45 天
        self.test_days = divide_days[1]  # 7*2 天
        self.history_length = history_length  # 6
        self.time_interval = time_interval  # 5 min  间隔一次数据

        self.one_day_length = int(24 * 60 / self.time_interval)

        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes)

        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]),
                                                               norm_dim=1)  # base , normalization

    def __len__(self):
        if self.train_mode == 'train':
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == 'test':
            return self.test_days * self.one_day_length
        else:
            raise ValueError('train mode:[{}] is not in defined'.format(self.train_mode))

    def __getitem__(self, index):
        '''
        :param index: int , range of dataset length [0, length-1]
        :return:
        '''
        if self.train_mode == 'train':
            index = index
        elif self.train_mode == 'test':
            index += self.train_days * self.one_day_length
        else:
            raise ValueError('train mode:[{}] is not in defined'.format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)
        data_x = LoadData.to_tensor(data_x)  # [N,H,D]
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # [N,1,D]

        return {'graph': LoadData.to_tensor(self.graph), 'flow_x': data_x, 'flow_y': data_y}

    @staticmethod
    def slice_data(data, history_length, index, train_mode):
        if train_mode == 'train':
            start_index = index
            end_index = index + history_length
        elif train_mode == 'test':
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError('train model:[{}] is not defined'.format(train_mode))

        data_x = data[:, start_index:end_index]
        data_y = data[:, end_index]
        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):
        '''
        :param data:np.array,original traffic data without normalization
        :param norm_dim:int, normalization dimension
        :return:
            norm_base,norm_data
        '''
        norm_base = LoadData.normalization_base(data, norm_dim)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)

        return norm_base, norm_data

    @staticmethod
    def normalization_base(data, norm_dim):
        '''
        :param data:np.array,original traffic data without normalization
        :param norm_dim:int, normalization dimension
        :return:
            max_data:np.array
            min_data:np.array
        '''
        max_data = np.max(data, norm_dim, keepdims=True)  # [N,T,D],norm = 1 -> [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base
        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        mid = min_data
        base = max_data - min_data
        recoverd_data = data * base + mid

        return recoverd_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':
    train_Data = LoadData(data_path=['..\\PeMS_04\\PeMS04.csv', '..\\PeMS_04\\PeMS04.npz'], num_nodes=307,
                          divide_days=[45, 14], time_interval=5, history_length=6, train_mode='train')
    print(train_Data)
    print(train_Data[0]['flow_x'].size())
    print(train_Data[0]['flow_y'].size())
