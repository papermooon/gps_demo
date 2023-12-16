import pandas as pd
import lightgbm as lgb
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer


def calculate_appearance(predict, label, proba):
    # TP = ((predict == label) & (label == 0)).sum()
    # TN = ((predict == label) & (label == 1)).sum()
    # FP = ((predict != label) & (label == 1)).sum()
    # FN = ((predict != label) & (label == 0)).sum()

    # acc = (TP + TN) / (TP + TN + FP + FN)
    # pre = (TP) / (TP + FP)
    # recall = (TP) / (TP + FN)
    # f1 = 2 * (pre * recall) / (pre + recall)
    # print(f"Accuracy: {acc} Precision: {pre} Recall: {recall} F1-SCORE: {f1}")

    prec, rec, f1, num = precision_recall_fscore_support(label, predict, average=None)
    print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f" % (prec[0], rec[0], f1[0]))
    print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f" % (prec[1], rec[1], f1[1]))
    micro_f1 = f1_score(label, predict, average='micro')
    print("Micro-Average F1 Score:", micro_f1)

    cm1 = confusion_matrix(label, predict)
    print(f"CM is: \n{cm1}")

    # fpr, tpr, thresholds = roc_curve(label, proba[:, 1])
    # auc_score = roc_auc_score(label, proba[:, 1])
    #
    # plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')  # 绘制ROC曲线，标注AUC的值
    # plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')  # 绘制随机分类器的ROC曲线
    #
    # plt.xlabel('False Positive Rate')  # x轴标签为FPR
    # plt.ylabel('True Positive Rate')  # y轴标签为TPR
    #
    # plt.title('ROC Curve')  # 设置标题
    # plt.legend()
    # plt.show()


raw_node = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
raw_class = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv')

raw_node.rename(columns={0: 'txId', 1: 'Times'}, inplace=True)
class_verify = raw_class[raw_class['class'] != "unknown"]
data = raw_node.merge(class_verify, left_on="txId", right_on="txId")

x = data.drop(["class", "txId", "Times"], axis=1)
x_local = x.drop(columns=[i for i in range(95, 167)], axis=1)
x_global = x.drop(columns=[i for i in range(2, 95)], axis=1)

y = data["class"]
y.replace('2', 0, inplace=True)
y_digit = pd.to_numeric(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_digit, test_size=0.3, shuffle=True, random_state=30,
                                                    stratify=y_digit)
model = lgb.LGBMClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

calculate_appearance(y_pred, y_test, y_proba)

# columns = x.columns.tolist()
# df = pd.DataFrame()
# df['feature name'] = columns
# df['importance'] = model.feature_importances_
# df = df.sort_values('importance', ascending=False)
# print(df['feature name'].tolist())
# df.plot.barh(x='feature name', figsize=(10, 12))
# plt.show()

print("-------------------------RF initializing---------------------------")
model3 = RandomForestClassifier(n_estimators=50, max_depth=100, max_features=50, random_state=15)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
y_proba = model3.predict_proba(x_test)

calculate_appearance(y_pred, y_test, y_proba)

print("-------------------------UMAP initializing---------------------------")
scaler = QuantileTransformer(random_state=0)  # 将数据映射到0-1之间的均匀分布（默认为均匀分布）
x = scaler.fit_transform(x)
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=5).fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(embedding, y_digit, test_size=0.3, shuffle=True, random_state=30,
                                                    stratify=y_digit)
model2 = lgb.LGBMClassifier()
model2.fit(x_train, y_train)

y_pred = model2.predict(x_test)
y_proba = model2.predict_proba(x_test)

calculate_appearance(y_pred, y_test, y_proba)
