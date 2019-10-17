nndl-book读书笔记-3.4-3.7

【数学知识补充：求多元空间中的某一点到某一个平面的距离】

1. Hessian矩阵：

   简单的说，它主要用来判定多元函数的极值，是多元函数在某一点的二阶偏导所组成的方阵。

   https://baike.baidu.com/item/黑塞矩阵/2248782?fr=aladdin

   多元函数Hessian矩阵为：

   ![1565578206489](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1565578206489.png)

2. 矩阵正定负定的充分必要条件：

   矩阵正定：所有的特征值均大于零

   矩阵负定：所有的特征值均小于零

   矩阵不定：特征值有大于零，也有小于零

   半正定矩阵：$X^TMX \geq0$, 直观上代表 一个向量经过它的变换后，向量与其本身的夹角小于等于90°。

3. 拉格朗日乘数法：

   给定而言函数z=f(x,y),和附加条件$\psi(x,y)=0$，为寻找z=f(x,y)在附加条件下的极值点，先做**拉格朗日函数**，$F(x,y,\lambda)=f(x,y)+\lambda \psi (x,y)$，其中$\lambda​$ 为参数。

4. 凸优化

##### 3.5 支持向量机SVM（support vector machine）

经典二分类算法：

![1565590179071](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1565590179071.png)

找到间隔$\gamma$越大，其分割平面对两个数据集的划分越稳定。

即目标为$max \ \gamma\     s.t. \      \frac {y^{n}(w^Tx^{(n)}+b)}{||w||}​$

令$||w|| \sdot \gamma =1$

则上式等价于：

$\max \    \frac{1}{||w||^2}  \    s.t.\     y^{(n)}(w^Tx^{(n)}+b)  \geq 1,\ \forall n​$

数据集中所有满足$y^{(n)}(w^Tx^{(n)}+b)  = 1$ 的点，称为支持向量。



SVM实现：

知识补充：

1. 凸优化函数包，以及凸优化标准化说明： https://blog.csdn.net/QW_sunny/article/details/79793889

   ![1565599052482](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1565599052482.png)

2. SVM决策函数为$f(x)=sgn(w*^Tx+b*)=sgn(\sum\lambda_n*y^{(n)}(x^{(n)}x+b*))$

3. 核函数：将样本空间从原始特征空间映射到更高维的空间，并解决原始特征空间中的线性不可分问题。

SVM算法实现：

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:50:45 2018

@author: cc
"""
import numpy as np
from numpy import linalg
import cvxopt

import pylab as pl
# 首先实现生成数据的类
class GenData(object):
    def liner_data(self):
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = [[0.8,0.6],[0.6,0.8]]
        #协方差矩阵
        X1 = np.random.multivariate_normal(mean1,cov,100)
        #np.random.multivariate_normal方法用于根据实际情况生成一个多元正态分布矩阵
        y1 = np.ones(100)
        X2 = np.random.multivariate_normal(mean2,cov,100)
        y2 = np.ones(100)*(-1)
        return X1,y1,X2,y2

    def gen_non_lin_separable_data(self):
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data(self):
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

# 实现SVM算法的类
class SVM(object):

    def kernelGen(self,x,y,kernel_type = 'liner',p=3):
        if kernel_type is 'liner':
            return np.dot(x,y)
        elif kernel_type is 'polynomial':
            return (1 + np.dot(x, y)) ** p
        else :
            return np.exp(-linalg.norm(x-y)**2 / (2 * (p ** 2)))

    def fit(self,X,y,kernel_type = 'liner',p=3,C = None):
        row,col = np.shape(X)
        K = np.zeros([row,row])
        for i in range (row):
            for j in range(row):
                K[i,j] = self.kernelGen(X[i],X[j],kernel_type,p)

        P = cvxopt.matrix(np.outer(y,y)*K)
        #np.outer表示的是两个向量相乘，拿第一个向量的元素分别与第二个向量所有元素相乘得到结果的一行。
        q = cvxopt.matrix(np.ones(row)*-1)
        A = cvxopt.matrix(y,(1,row))
        b = cvxopt.matrix(0.0)
        print("cvxopt")
        print("P",np.shape(P))
        print("q",np.shape(q))
        print("A",np.shape(A))
        print("b",np.shape(b))
        if C is None:
            G = cvxopt.matrix(np.diag(np.ones(row) * -1))
            h = cvxopt.matrix(np.zeros(row))
            print("G",np.shape(G))
            print("h",np.shape(h))
        else :

            C = float(C)
            tmp1 = np.diag(np.ones(row) * -1)
            tmp2 = np.identity(row)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(row)
            tmp2 = np.ones(row) *C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        #cvxopt ConvexOptimization凸优化模块
        alafa = np.ravel(solution['x'])#
        # print("alafa",alafa)
        sv = alafa>1e-5
        # print("sv",sv)
        index = np.arange(row)[sv]
        self.alafa = alafa[sv]
        self.sv_x = X[sv]  # sv's data
        self.sv_y = y[sv]  # sv's labels
        print("%d support vectors out of %d points" % (len(self.alafa), row))

        self.b = 0
        for n in range(len(self.alafa)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alafa * self.sv_y * K[index[n],sv])
        self.b /= len(self.alafa)

    def project(self,X,kernel_type='liner',p=3):
        #核函数映射
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alafa, sv_y, sv_x in zip(self.alafa, self.sv_y, self.sv_x):
                s += alafa * sv_y * self.kernelGen(X[i],sv_x,kernel_type,p)
                #计算\sum (lamda*y*K(x,z))
            y_predict[i] = s
        return y_predict + self.b

    def predict(self,X,kernel_type,p):
        return np.sign(self.project(X,kernel_type,p))

class TestAndTrain(object):

    def split_train(self,X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(self,X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_contour(self,X1_train, X2_train, clf,kernel_type='liner',p=3):
        # 作training sample数据点的图
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        # 做support vectors 的图
        pl.scatter(clf.sv_x[:,0], clf.sv_x[:,1], s=100, c="g")
        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X,kernel_type,p).reshape(X1.shape)
        # pl.contour做等值线图
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def trainSVM (self,X_train,y_train,X_test,y_test,kernel_type = 'liner',p=3,C=None):
        clf = SVM()
        clf.fit(X_train, y_train,kernel_type,p,C)
        y_predict = clf.predict(X_test,kernel_type,p)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        self.plot_contour(X_train[y_train==1], X_train[y_train==-1], clf,kernel_type,p)


if __name__ == "__main__":

    def liner_test():
        gen = GenData()
        tt = TestAndTrain()
        X1, y1, X2, y2 = gen.liner_data()
        X_train, y_train = tt.split_train(X1, y1, X2, y2)
        X_test, y_test = tt.split_test(X1, y1, X2, y2)
        tt.trainSVM(X_train,y_train,X_test,y_test,'liner')

    def soft_test():
        print("soft_test")
        gen = GenData()
        tt = TestAndTrain()
        X1, y1, X2, y2 = gen.gen_lin_separable_overlap_data()
        X_train, y_train = tt.split_train(X1, y1, X2, y2)
        X_test, y_test = tt.split_test(X1, y1, X2, y2)
        tt.trainSVM(X_train,y_train,X_test,y_test,'liner',3,0.5)


    def nonliner_test():
        print("nonliner_test")
        gen = GenData()
        tt = TestAndTrain()
        X1, y1, X2, y2 = gen.gen_non_lin_separable_data()
        X_train, y_train = tt.split_train(X1, y1, X2, y2)
        X_test, y_test = tt.split_test(X1, y1, X2, y2)
        tt.trainSVM(X_train,y_train,X_test,y_test,'gaussion',5)
    liner_test()
    #asoft_test()
    #nonliner_test()


```

