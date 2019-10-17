nndl-book读书笔记-1.0-2.4

## 机器学习

##### 1. 目标

机器学习的目标是找到一个模型来近似真实映射函数g(x)或真实条件概率分布。

##### 2. 模型

**线性**：$f(x,\theta)=w^Tx+b​$

参数θ包含了权重向量w和偏置b.

**非线性**：$f(x,\theta)=w^T \phi(x)+b$

参数$\phi(x)=[\phi_1(x), \phi_2(x),..., \phi_K(x)]​$为K个非线性基函数组成的向量，参数θ包含权重向量w和偏置b。

##### 3.损失函数

（1） 0-1损失函数

（2）$L(y, f(x, \theta))=1/2  (y-f(x, \theta))^2$ 

（3）交叉熵损失函数

$f_c=-\sum y_c log f_c(x,\theta) ​$

（对数损失函数是交叉熵的二元情况）

（4）Hinge损失函数: $L(Y,f(x, \theta))=max(0,1-y f(x,\theta))$

##### 4.风险最小化

经验风险empirical risk, 即在训练集上的平均损失。

$R_D^{emp}(\theta)=1/N \sum L(y^{(n)},f(x^{(n)},\theta))​$ 

经验风险最小化原则很容易导致模型在训练集上错误率很低，但是在未知数据上错误率很高，即**过拟合**（overfitting）.

过拟合产生原因：由于训练数据少和噪声以及模型能力强等原因造成。

解决方法：

（1）在经验风险的基础上再引入参数的正则化regularization, 限制模型能力。

![1563939759603](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1563939759603.png)



【

大数定律：

![1563939544301](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1563939544301.png)

】

欠拟合underfitting

##### 5. 优化算法

参数与超参数

$f(x, \theta)$ 中的$\theta$ 称为模型的参数，可以通过优化算法进行学习；

用于定义模型结构或优化策略的，称为超参数hyper-parameter。

（1）梯度下降算法

$\theta_{t+1}=\theta_{t}-\alpha \frac{\partial R_D(\theta)}{\partial \theta}$ 

$\alpha$为学习率

针对梯度下降算法，解决过拟合的方法还有提前停止。

（2）BGD 批量梯度下降算法

批量梯度下降相当于是从真实数据分布中采集 *N* 个样本，并由它们计算出来的经验风险的梯度来近似期望风险的梯度。

（3）SGD随机梯度下降算法

每次迭代时只采集一个样本，计算这个样本损失函数的梯度并更新参数，即随机梯度下降法。

（4）mini-BGD 小批量梯度下降算法

![1563946854539](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1563946854539.png)





（*K* 通常不会设置很大，一般在 1 *∼* 100 之间。在实际应用中为了提高计算效率，通常设置为2的*n*次方。）

##### 6. 线性回归

机器学习任务可以分为两类，一类是样本的特征向量 **x** 和标签 *y* 之间如果存在未知的函数关系 *y* = *h*(**x**)，另一类是条件概率 *p*(*y**|***x) 服从某个未知分布。

（1）最小二乘法Least Square Estimation线性回归代码如下：

```python
import numpy as np
SIZE_COL = 8
SIZE_ROW = 1
def function(x:float)->float:
    return 2*x-1

def generate():#产生y=2x-1的随机输入，输出
    np.random.seed(1)
    x_input = np.random.uniform(0, 10, size=[SIZE_ROW, SIZE_COL])
    # print(x_input)
    y_output = np.random.uniform(-1, 1, size=[SIZE_ROW, SIZE_COL])
    # print(x_input)
    for i in range(SIZE_ROW):
        for j in range(SIZE_COL):
            y_output[i][j] = y_output[i][j] + function(x_input[i][j])
            # print(y_output[i][j])
    return x_input,y_output

def add_bias(x_input,row,col):
    add_row=np.ones([row,col])
    x_input=np.row_stack((x_input,add_row))
    return x_input

def train(x_input,y_output):
    w = np.zeros([2, 1])  # y=w1x+w2
    x_bias_input = add_bias(x_input, SIZE_ROW, SIZE_COL)

    result = np.dot(x_bias_input, x_bias_input.T)
    reverse_x = np.linalg.inv(result)
    result2 = np.dot(reverse_x, x_bias_input)
    result3 = np.dot(result2, y_output.T)
    return result3

x_input,y_output=generate()
w=train(x_input,y_output)
print("函数关系为：y=",w[0][0],"x+",w[1][0])

```

（2）通过建模条件概率*p*(*y**|***x)的角度来进行参数估计线性回归参数

【

似然$p(x|w)​$和概率$p(x|w)​$之间的区别在于：概 率p*(*x**|**w*)是描述固定参数*w时，随机变量 *x* 的分布情况，而似然*p*(*x**|**w*)则是描述已知随机变量 *x* 时，不同的参数 *w*对其分布的影响

】

最大似然估计（Maximum Likelihood Estimate，MLE）是指找到一组参数w**使得似然函数*p*(**y***|**X,* **w***, σ*)最大，等价于对数似然函数log *p*(**y***|**X,* **w***, σ*)最大。

关键概念掌握：对数似然函数p40页

**最大后验估计**：最大后验估计（Maximum A Posteriori Estimation，MAP）是指最优参数为后验分布p(w|X, y, ν, σ)中概率密度最高的参数w。

$w^{MAP}=arg _w max p(y|X,w,\sigma) p(w|v)$ 

![1564025652147](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564025652147.png)



##### 7. 偏差-方差分解

bias偏差描述的是算法的预测的平均值和真实值的关系（可以想象成算法的拟合能力如何），variance方差描述的是同一个算法在不同数据集上的预测值和所有数据集上的平均预测值之间的关系。

对于单个样本x，不同训练集*D*得到模型$f_D(x)$和最优模型$f^*(x)$的上的期望差距为:

![1564039193657](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564039193657.png)

因此期望错误可以分解为：

$R(f)=(bias)^2+variance+\varepsilon $ 

模型的拟合能力变强，偏差减少而方差增大，而导致过拟合。以结构错误最小化为例，我们可以调整正则化系数*λ*来控制模型的复杂度模型复杂度会降低，可以有效地减少方差，避免过 结构错误最小化 拟合，但偏差会上升。