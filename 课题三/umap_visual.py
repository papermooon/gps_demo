import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from umap import UMAP

raw_node = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
raw_class = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
raw_egdes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')

raw_node.rename(columns={0: 'txId', 1: 'Times'}, inplace=True)

class_verify = raw_class[raw_class['class'] != "unknown"]
data = raw_node.merge(class_verify, left_on="txId", right_on="txId")

# raw_class['class'].replace('unknown', '3', inplace=True)
# data = raw_node.merge(raw_class, left_on="txId", right_on="txId")

x = data.drop(["class", "txId", "Times"], axis=1)
y = data["class"]
y_digit = pd.to_numeric(y)

scaler1 = StandardScaler()
scaler2 = MinMaxScaler(feature_range=(-1, 1))
scaler3 = QuantileTransformer(random_state=0)  # 将数据映射到0-1之间的均匀分布（默认为均匀分布）
scaler4 = QuantileTransformer(random_state=0,
                              n_quantiles=2000,  # default =1000
                              output_distribution='normal')

x.columns = x.columns.astype(float)
x = scaler3.fit_transform(x)

embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=20).fit_transform(x)
ax = plt.axes(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=y_digit / 3, cmap="viridis")
# plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.to_numeric(y) / 2, cmap="viridis", s=5)

# x_train, x_test, y_train, y_test = train_test_split(x, y_digit, test_size=0.25, shuffle=False)
# reducer = UMAP(n_components=20, n_epochs=1000, min_dist=0.1)
# x_train_res = reducer.fit_transform(x_train, y_train)
# x_test_res = reducer.transform(x_test)
# ax = plt.axes(projection='3d')
# ax.scatter(x_test_res[:, 0], x_test_res[:, 1], x_test_res[:, 2], c=y_test / 3, cmap="viridis")

plt.show()
