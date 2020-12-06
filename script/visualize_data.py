import numpy as np
import matplotlib.pyplot as plt

def get_flow(filename):
    flow_data = np.load(filename)
    # print([key for key in flow_data.keys()])   'data'
    # flow_data = flow_data['data'].transpose([1,0,2])[:,:,0][:,:,np.newaxis]   #[N,T,D] D=1  匹配维度
    flow_data = flow_data['data'].transpose([1, 0, 2])
    return flow_data

if __name__ == '__main__':
    traffic_data = get_flow('../PeMS_04/PeMS04.npz')
    node_id = range(5)   #5个不同地区观察点
    print(traffic_data.shape)    #(16992, 307, 3) 转变后 (307, 16992, 3)
    for i in node_id:
        plt.plot(traffic_data[i,:24*12,0])  #一天24小时，每5分钟采集一次数据，每小时12个数据
        plt.show()   #蓝色交通时间流量   绿色速度