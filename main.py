# -*- coding:utf-8 -*-
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.optimize import leastsq

lineExtraX = 1


##需要拟合的函数func :指定函数的形状
def func(p, x):
    k, b = p
    return k * x + b


##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p, x, y):
    return func(p, x) - y


# X = np.array(
#     [[1, 1], [2, 2], [3, 2], [1, 1.5], [2, 2.5], [3.5, 2], [1, 1.5],
#      [0.5, 2], [3, 3.5], [5.5, 1.5], [6.5, 2.5], [3.5, 4.5]])


##python自带的迭代器模块
##产生随机数据的中心
centers = [[1, 1], [2, 2], [3, 5], [5, 2], [6, 6]]
# centers = [[1, 1], [2, 2]]
##产生的数据个数
n_samples = 200
##生产数据
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.5, random_state=0)

ms = MeanShift(bandwidth=0.5, bin_seeding=True)
##训练数据
ms.fit(X)
##每个点的标签
labels = ms.labels_
# print(labels)
##簇中心的点的集合
cluster_centers = ms.cluster_centers_
print('集群中心:', cluster_centers)

##总共的标签分类
labels_unique = np.unique(labels)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("集群中心数量: %d" % n_clusters_)

sum_yx = 0
sum_x2 = 0
sum_delta = 0

minX = cluster_centers[0][0]
maxX = cluster_centers[0][0]
x_bar = np.mean(cluster_centers[0])
m = len(cluster_centers)
# print(m)
for i in range(m):
    x = cluster_centers[i][0]
    y = cluster_centers[i][1]
    minX = min(minX, x)
    maxX = max(maxX, x)
    sum_yx += y * (x - x_bar)
    sum_x2 += x ** 2
w = sum_yx / (sum_x2 - m * (x_bar ** 2))

for i in range(m):
    x = cluster_centers[i][0]
    y = cluster_centers[i][1]
    sum_delta += (y - w * x)
b = sum_delta / m

print("求解的拟合直线为:")
print("y=" + str(round(w, 2)) + "x+" + str(round(b, 2)))

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
             markeredgecolor='k', markersize=10)
plt.title('Result')
x = np.linspace(minX - lineExtraX, maxX + lineExtraX, 100)
y = w * x + b  ##函数式
plt.plot(x, y, color="red", label="拟合直线", linewidth=2)
plt.show()
