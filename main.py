# -*- coding:utf-8 -*-
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# X = np.array(
#     [[1, 1], [2, 2], [3, 2], [1, 1.5], [2, 2.5], [3.5, 2], [1, 1.5],
#      [0.5, 2], [3, 3.5], [5.5, 1.5], [6.5, 2.5], [3.5, 4.5]])


##python自带的迭代器模块
##产生随机数据的中心
centers = [[1, 1], [2, 2], [3, 5]]
##产生的数据个数
n_samples = 200
##生产数据
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.8, random_state=0)

ms = MeanShift(bandwidth=0.9, bin_seeding=True)
##训练数据
ms.fit(X)
##每个点的标签
labels = ms.labels_
# print(labels)
##簇中心的点的集合
cluster_centers = ms.cluster_centers_
print('cluster_centers:', cluster_centers)
##总共的标签分类
labels_unique = np.unique(labels)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

##绘图
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    ##根据lables中的值是否等于k，重新组成一个True、False的数组
    my_members = labels == k
    cluster_center = cluster_centers[k]
    ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
