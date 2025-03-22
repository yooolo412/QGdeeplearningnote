import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
#使用黑体汉字
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #解决负号显示问题
from networkx.algorithms.traversal import dfs_successors

#读取csv文件
boston_df = pd.read_csv("data/boston.csv")
#显示前五行
#print(df.head())

#拼接特征data和预测目标target
#complet_df = np.column_stack([df,])

#数据预处理
# 检查数据是否有缺失值
print(boston_df.isnull().sum())
# 查看数据信息
print(boston_df.describe())
#绘制直方图
boston_df.hist(bins=20,figsize=(20,15))
plt.show()

# 计算相关性矩阵
correlation = boston_df.corr()
# 按照与房价的相关性排序
print(correlation['MEDV'].sort_values(ascending=False))


#分析房间数与房价关系
#异常值处理,查看房间数异常值，房间数>8一般比较少，可以直接用8替换
boston_df.boxplot(column='RM')
plt.show()
boston_df.loc[boston_df['RM']>8,'RM']=8

plt.figure(figsize=(15,10))
plt.scatter(boston_df['RM'],boston_df['MEDV'],alpha=0.5)
plt.xlabel('平均房间数（RM）')
plt.ylabel('房价中位数（MEDV）')
plt.title('RM vs MEDV')
plt.show()

#分析低收入人群比例与房价关系
plt.figure(figsize=(15,10))
plt.scatter(boston_df['LSTAT'],boston_df['MEDV'],alpha=0.5)
plt.xlabel('低收入人群比例（LSTAT）')
plt.ylabel('房价中位数（MEDV）')
plt.title('LSTAT vs MEDV')
plt.show()

