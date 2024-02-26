import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
# import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import random
import torch
import networkx as nx
import pymetis
import numpy as np
from torch_geometric.data import Data
import dgl


def calculate_appearance(predict, label):
    prec, rec, f1, num = precision_recall_fscore_support(label, predict, average=None)
    print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f" % (prec[0], rec[0], f1[0]))
    print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f" % (prec[1], rec[1], f1[1]))
    micro_f1 = f1_score(label, predict, average='micro')
    print("Micro-Average F1 Score:", micro_f1)
    cm1 = confusion_matrix(label, predict)
    print(f"CM is: \n{cm1}")


# 1.降维
def dim_reduction():
    data_raw = pd.read_csv('./feat_label.csv')
    data = data_raw[data_raw['class'] != 2]
    x = data.drop(["class", "txId", "Times"], axis=1)
    y = data["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=30,
                                                        stratify=y)
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    calculate_appearance(y_pred, y_test)

    from sklearn.feature_selection import SelectFromModel
    new_model = SelectFromModel(model, prefit=True)  # 已经fit过了，所以必须带上参数prefit

    info_df_1 = data_raw[["txId", "Times"]]
    info_df_2 = data_raw["class"]
    x = data_raw.drop(["class", "txId", "Times"], axis=1)

    x_new = new_model.transform(x)
    x_new = pd.DataFrame(x_new)

    x_new = pd.concat([info_df_1, x_new], axis=1)
    x_new = pd.concat([x_new, info_df_2], axis=1)
    print(x_new.head())
    print(data_raw.head())
    print(x_new.tail())
    print(data_raw.tail())
    x_new.to_csv('feat_label_reduc.csv', index=False)


# 2.离散化
def digitalize(param_init):
    data_raw = pd.read_csv('./feat_label_reduc.csv')
    node_feature = data_raw.drop(["class", "txId", "Times"], axis=1)
    if param_init:
        for i in range(0, 57):
            data = node_feature[str(i)]
            data_len = data.shape[0]
            data_reshape = data.values.reshape((data.shape[0], 1))

            X = range(2, 20)
            SSE = []  # 存放每次结果的误差平方和

            for k in range(2, 20):
                estimator = KMeans(n_clusters=k, random_state=0, n_init='auto')
                estimator.fit_predict(data_reshape)
                SSE.append(estimator.inertia_ / data_len)  # estimator.inertia_获取聚类准则的总和
            plt.xlabel('k-feat-' + str(i))
            plt.xticks(range(1, 25))
            plt.ylabel('SSE')
            plt.plot(X, SSE, 'o-')
            plt.savefig('./feat_work/feat' + str(i) + '.png')
            plt.show()
    hyper_params = [9, 6, 6, 10, 9, 9, 5, 10, 8, 5, 5, 7, 8, 9, 7, 7, 7, 8, 9, 8, 8, 7, 5, 6, 7, 4, 7, 9, 9, 8, 9,
                    9, 12, 10, 11, 11, 10, 10, 10, 10, 5, 5, 8, 10, 9, 11, 6, 8, 10, 7, 10, 10, 8, 10, 12, 9, 5]
    print(len(hyper_params))
    for i in range(0, 57):
        data = data_raw[str(i)]
        data_reshape = data.values.reshape((data.shape[0], 1))
        estimator = KMeans(n_clusters=hyper_params[i], random_state=0, n_init='auto')
        res = estimator.fit_predict(data_reshape)
        data_raw[str(i)] = res
    data_raw.to_csv('reduced_digit_fit_label.csv', index=False)


# 3.时间片划分子图
def sub_graph_generate():
    data_raw = pd.read_csv('./reduced_digit_fit_label.csv')
    for i in range(1, 50):
        filename = 'sub_' + str(i) + '.csv'
        tmp = data_raw[data_raw['Times'] == i]
        tmp.to_csv('./reduced_subs/' + filename, index=False)


# 4. 进一步拆分每张子图
def refine_sub_graph():
    refine_num = 10
    raw_edges = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    dirname = './subs/'
    max_rate = 0
    min_rate = 100
    for i in range(1, 50):
        filename = 'sub_' + str(i) + '.csv'
        data = pd.read_csv(dirname + filename)
        sample = data[data['class'] == 1]
        rate = len(sample) / len(data) * 100
        print("lable:", len(sample), " sum:", len(data), " rate:", rate, "%")
        max_rate = max(rate, max_rate)
        min_rate = min(rate, min_rate)

        # 异常节点保留，正常节点划分成num份
        partion_node = data[data['class'] != 1]

        # 按txId排序
        sub_graph = partion_node.sort_values('txId').reset_index(drop=True)

        # 注意！异常节点的点和边先没管！
        nodes = sub_graph['txId'].values
        sample_nodes = sample['txId'].values

        print(nodes)
        index_list = []
        for node in nodes:
            index_list += raw_edges[raw_edges['txId1'] == node].index.tolist()
            index_list += raw_edges[raw_edges['txId2'] == node].index.tolist()
        index_list = list(set(index_list))
        print(len(index_list))

        index_list = pd.Index(index_list)
        sub_edges = raw_edges.loc[index_list]

        map_id = {j: i for i, j in enumerate(nodes)}
        sub_edges.txId1 = sub_edges.txId1.map(map_id)
        sub_edges.txId2 = sub_edges.txId2.map(map_id)
        # 因为异常节点被排除了，所以这里会有部分边映射为NaN
        # print(sub_edges.isnull().values.any())
        sub_edges.dropna(axis=0, how='any', inplace=True)

        edges_list = []
        for tup in zip(sub_edges['txId1'], sub_edges['txId2']):
            ele = [int(tup[0]), int(tup[1])]
            edges_list.append(ele)

        node_list = list(range(len(nodes)))
        G = nx.DiGraph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edges_list)
        edgecuts, parts = pymetis.part_graph(6, G)

        print(G)
        return
        # G_dgl = dgl.from_networkx(G)
        # print(G_dgl)
        # dgl.distributed.partition_graph(G_dgl, 'test', 10, num_hops=1, part_method='metis',
        #                                 out_path='output/', reshuffle=True,
        #                                 balance_ntypes=g.ndata['train_mask'],
        #                                 balance_edges=True)
        # print(G_dgl.nodes())
        # print(G_dgl.edges())

        # # 获得 Metis 划分结果
        # (edgecuts, parts) = pymetis.part_graph(4, G)
        # print(edgecuts)
        # print(parts)
        # print(len(parts))

        # # 将划分结果应用到 PyG 图对象
        # x = torch.ones(G.number_of_nodes(), 1)  # 节点特征
        # edge_index = torch.tensor(list(G.edges)).t().contiguous()  # 边索引
        # # 根据 Metis 划分结果创建节点划分的张量
        # partition_tensor = torch.tensor(parts, dtype=torch.long)
        # # 创建 PyG 图对象
        # pyg_graph = Data(x=x, edge_index=edge_index, part=partition_tensor)
        # print(pyg_graph)

        # 重写各节点id，重写连边

        node_feature = sub_graph.drop(["class", "txId", "Times"], axis=1)
        data_x = torch.tensor(np.array(node_feature.values), dtype=torch.long)

        node_label = sub_graph['class']
        data_y = torch.tensor(node_label, dtype=torch.long)

        edge_index = torch.tensor(np.array(sub_edges.values), dtype=torch.long).T

        partion_G_pyG = Data(x=data_x, edge_index=edge_index, y=data_y)
        print(partion_G_pyG)
        from torch_geometric.utils.convert import to_networkx
        partion_G_nx = to_networkx(partion_G_pyG, to_undirected=False, node_attrs=['x'])
        print(partion_G_nx)

        # print(f'节点名：{partion_G_nx.nodes}')
        # print(f'边的节点对：{partion_G_nx.edges}')
        # print('每个节点的属性：')
        # for node in partion_G_nx.nodes(data=True):
        #     print(node)
        # print('每条边的属性：')
        # for edge in partion_G_nx.edges(data=True):
        #     print(edge)

        #
        # # 创建一个简单的图
        # G = nx.cycle_graph(8)
        # nx.draw(G, node_size=30)
        # plt.show()
        # # 获得 Metis 划分结果
        # (edgecuts, parts) = metis.part_graph(3, G)
        # print(edgecuts)
        # print(parts)
        # # 将划分结果应用到 PyG 图对象
        # x = torch.ones(G.number_of_nodes(), 1)  # 节点特征
        # edge_index = torch.tensor(list(G.edges)).t().contiguous()  # 边索引
        # # 根据 Metis 划分结果创建节点划分的张量
        # partition_tensor = torch.tensor(parts, dtype=torch.long)
        # # 创建 PyG 图对象
        # pyg_graph = Data(x=x, edge_index=edge_index, part=partition_tensor)
        # print(pyg_graph)

        break
    print(max_rate, min_rate)
    pass


# 4.1 直接什么都不管，先划分子图。
def dumb_sub_graph():
    refine_num = 10
    raw_edges = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    dirname = './subs/'

    for i in range(1, 50):
        filename = 'sub_' + str(i) + '.csv'
        data = pd.read_csv(dirname + filename)

        sub_graph = data.sort_values('txId').reset_index(drop=True)

        nodes = sub_graph['txId'].values
        index_list = []
        for node in nodes:
            index_list += raw_edges[raw_edges['txId1'] == node].index.tolist()
            index_list += raw_edges[raw_edges['txId2'] == node].index.tolist()
        index_list = list(set(index_list))
        index_list = pd.Index(index_list)

        sub_edges = raw_edges.loc[index_list]
        map_id = {j: i for i, j in enumerate(nodes)}
        sub_edges.txId1 = sub_edges.txId1.map(map_id)
        sub_edges.txId2 = sub_edges.txId2.map(map_id)
        sub_edges.dropna(axis=0, how='any', inplace=True)

        edges_list = []
        for tup in zip(sub_edges['txId1'], sub_edges['txId2']):
            ele = [int(tup[0]), int(tup[1])]
            edges_list.append(ele)
        node_list = list(range(len(nodes)))

        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edges_list)
        edgecuts, parts = pymetis.part_graph(refine_num, G)

        datalist = []
        for part in range(refine_num):
            indices_of_part = [index for index, value in enumerate(parts) if value == part]
            node_indices = pd.Index(indices_of_part)
            part_node = sub_graph.loc[node_indices]
            part_edge = sub_edges[sub_edges['txId1'].isin(indices_of_part) & sub_edges['txId2'].isin(indices_of_part)]

            # 重写各节点id，重写连边
            map_tmp = {j: i for i, j in enumerate(indices_of_part)}
            part_edge['txId1'] = part_edge.txId1.map(map_tmp)
            part_edge['txId2'] = part_edge.txId2.map(map_tmp)

            node_feature = part_node.drop(["class", "txId", "Times"], axis=1)
            data_x = torch.tensor(np.array(node_feature.values), dtype=torch.long)
            node_label = part_node['class']
            data_y = torch.tensor(np.array(node_label.values), dtype=torch.long)
            edge_index = torch.tensor(np.array(part_edge.values), dtype=torch.long).T

            part_data = Data(x=data_x, edge_index=edge_index, y=data_y)
            datalist.append(part_data)

        torch.save(datalist, dirname + 'graph_data' + str(i) + '.pt')
        # # 加载图数据
        # loaded_data = torch.load('graph_data.pt')
        # print(loaded_data)

    return


# dim_reduction()
# digitalize(param_init=False)
# sub_graph_generate()
# refine_sub_graph()
dumb_sub_graph()
