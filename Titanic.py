import warnings
warnings.filterwarnings("ignore")   # 忽略警告信息

import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# 加载数据
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
combine = [train_df, test_df]


# 查看所有特征名、数据
# print(train_df.columns.values)  
# print(train_df.head())
# print(train_df.tail())


# 查看缺失值
# print(train_df.isnull().sum())
# print(test_df.isnull().sum())


# 预览每个变量的基本信息
# print(train_df.info())
# print(test_df.info())


# 查看数值分布
# print(round(train_df.describe(percentiles = [.5, .6, .7, .8, .9, .99])), 2)


# 查看样本中的特征分布
# print(train_df.describe(include = ['O']))   # 查询字符串列的特征分布


# 特征工程
'''
1. Age 年龄与存活相关
2. Embarked 登船港口与存活或其他特征相关
3. Ticket 票号有较高重复率 
4. Cabin 客舱号 缺失值过多 丢弃
5. PassengerId 乘客编号无用 丢弃
6. Name 名字 特征不规范 可能丢弃
'''

# .groupby(分类列)进行数据分组, 并求每组的均值, 再排降序
pcl_surv_feat = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_surv_feat = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sib_surv_feat = train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
par_surv_feat = train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# print(pcl_surv_feat)
# print('-'*40)
# print(sex_surv_feat)
# print('-'*40)
# print(sib_surv_feat)
# print('-'*40)
# print(par_surv_feat)

'''
从数据中可以看出 Pclass:1、female、SibSp:1、Parch:3 存活率更高
SibSp和Parch这两个特征类似, 可以选择合并这两个特征创造一个新特征
'''


# 可视化数据分析, 需要用到seaborn这个库
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20, color='orange')


# 有序分类特征 票价与存活相关性
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
# grid.add_legend()


# 无序分类特征 港口与存活相关性
# grid = sns.FacetGrid(train_df, col='Embarked')
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()



# 数值特征 Fare与存活相关性, 并将Embarked(非数值)、Fare(连续数值)、Survived(二分类数值)相关联
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0:'b', 1:'r'})
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()



# 清洗数据
# 删除无用特征 Ticket、Cabin
# print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
# print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# 用新特征 Title去替代 Name、PassengerId
# print(train_df['Name'].head(10))
for data in combine:
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)

tab = pd.crosstab(train_df['Title'], train_df['Sex']).sort_values(by='female', ascending=False)

# grid = sns.FacetGrid(train_df, col='Title', hue='Survived')
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()


# 删去或合并某些 Title
for data in combine:
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

t_s_tab = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(t_s_tab)


# 特征序数化
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, "Rare":5}
for data in combine:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    data['Sex'] = data['Sex'].map({'female':1, 'male':0}).astype(int)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# print(train_df.shape, test_df.shape)


# 处理年龄特征的缺失值
'''
1. 用均值或中位数填补
2. 联合其他相关特征填补(这里Age、Sex、Pclass之间具有相关性)
'''
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Sex')
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

# 通过 Sex(0、1)和 Pclass(1、2、3)来预测 Age值
# loc：根据 DataFrame的具体标签选取行列, 同样是先行标签, 后列标签
guess_age = np.zeros((2, 3))

for data in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = data[(data['Sex'] == i) & (data['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_age[i, j] = int(age_guess/0.5 + 0.5) * 0.5    # 四舍五入
    
    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[(data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1), 'Age'] = guess_age[i, j]
    
    data['Age'] = data['Age'].astype(int)


# 分类模型要对连续变量离散化, 以使模型更稳点, 预防过拟合
# 给年龄分段处理, pd.cut()进行等距分箱
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
a_t = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for data in combine:
    data.loc[data['Age'] < 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]


# 结合 SibSp和 Parch特征
for data in combine:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

F_t = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for data in combine:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

A_t = train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# 舍弃Parch、SibSp、FamilySize特征, 只用IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


# 联合 Age*Pclass
for data in combine:
    data['Age_Pclass'] = data['Age'] * data['Pclass']

AP_T = train_df[['Age_Pclass', 'Survived']].groupby(['Age_Pclass'], as_index=False).mean()



# 填补 Embarked
freq_port = train_df.Embarked.dropna().mode()[0]
for data in combine:
    data['Embarked'] = data['Embarked'].fillna(freq_port)

E_t = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for data in combine:
    data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)


# 票价处理
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
# plt.hist(train_df['Fare'])
# plt.show()

# 等频分箱
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
FB_t = train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived', ascending=True)

for data in combine:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
# print(train_df.head())



# 定义模型
X_train, Y_train = train_df.drop(['Survived'], axis=1), train_df['Survived']
X_test = test_df.drop(['PassengerId'], axis=1).copy()
# print(X_train.shape, Y_train.shape, X_test.shape)

# 逻辑回归
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train)*100, 2)

# SVM
svm = SVC()
svm.fit(X_train, Y_train)
acc_svm = round(svm.score(X_train, Y_train)*100, 2)

# KNN, n_neighbors是选择临近的 n个点进行计算
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)

# 朴素贝叶斯分类
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)

# 感知机
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)

# 线性SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100, 2)

# 决策树
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)

# sgd
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)

# random forest, n_estimators：随机森林中决策树的个数, 默认为 100
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)



# 模型评估
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svm, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]
})
m_t = models.sort_values(by='Score', ascending=False)
# print(m_t)



# 提交
submission = pd.DataFrame({
    "PassengerId": test_df['PassengerId'],
    "Survived": Y_pred
})
submission.to_csv('submission.csv', index=False)


# 计算存活率
submission_df = pd.read_csv("submission.csv")
surv = (submission_df['survived'] == 1).sum() / len(submission_df)
print('survived_rate: ', surv*100)
