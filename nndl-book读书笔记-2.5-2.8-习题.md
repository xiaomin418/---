nndl-book读书笔记-2.5-2.8-习题

1.机器学习算法类型

机器学习分类：线性/非线性，统计方法/非统计方法。

一般来说：按照训练样本**提供的信息**以及**反馈方式**不同，将机器学习分为：

**监督学习**：通过建模样本的特征x和标签y之间的关系，并且训练集中每个样本都有标签。

（1）回归：标签y是连续值，$f(x,\theta)​$也连续。

（2）分类：标签y是离散的类别。

（3）结构化学习：输出是结构化的对象。

**无监督学习**：从不包含目标标签的训练样本中自动学习到一些有价值的信息。

典型：聚类，密度估计，特征学习，降维。

2.数据特征表示

图像特征，文本特征，表示学习

词袋模型：

比如两个文本：“<u>我</u> <u>喜欢</u> <u>读书</u>”, “<u>我</u> <u>讨厌</u> <u>读书</u>”。

二元特征（即两个词的组合特征）：“$我”，“我喜欢”，“我讨厌”，“喜欢读书”，“讨厌读书”，“读书#”, BoW模型表示为：

$v_1=[1\ 1\ 0\ 1\ 0\ 1 ]^T​$ 

$v_2=[1\ 0\ 1\ 0\ 1\ 1 ]^T​$  

3.特征选择：选择原始特征集合的一个有效子集，使得基于这个特征子集训练出来的模型准确率最高。

**深度学习：**将特征学习的表示学习和机器学习的预测学习有机的统一到一个模型中，建立一个端到端的学习算法，可以有效的避免它们之间准则的不一致性，这种表示学习方法就称为深度学习。

3. 评价指标

   准确率accuracy: $ACC=\frac {1}{N}\sum I(y^{(n)}=y*^{(n)})​$

   错误率Error rate: $\epsilon = 1-ACC​$

TP, FN, FP, TN

#### 2.8 理论和定理

PAC学习理论：帮助分析一个机器学习方法在什么条件下可以学习到一个近似正确的分类器。

$n(\epsilon, \delta)=\frac{1}{2\delta^2}(ln|F|+ln\frac{2}{\delta})$ 

NFL（no free lunch theorem）

对于基于地带的最优化算法，不存在某种算法对所有问题都有效。

丑小鸭定理：世界上不存在相似性的客观标准，一切相似性的标准都是主观的。

Occam's razor: 简单的模型泛化能力好，如果有两个性能详尽的额模型，我们应该选择更简单的模型。

$max_flog p(f|D)=max_flog p(D|f) + log p(f)=min_f - log p(D|f)-log p(f)​$ 其中*]* log *p*(*f*)和*¼* log *p*(*D|*f*)可以分别看作是模型*f* 的编码长度和在该模型数据集*D*的编码长度。也就是说，我们不但要使得模型*f* 可以编码数据集*D*，要使得模型*f* 尽可能简单。

 归纳偏置：在贝叶斯中称为先验。

#### 习题：

2-1 分析为什么平方损失函数不适用于分类问题？

![1564986461759](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564986461759.png)

![1564986473659](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564986473659.png)

![1564986488887](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564986488887.png)

![1564986529406](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564986529406.png)

![1564986540349](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564986540349.png)

2-2 在线性回归中，如果我们给每个样本$(x^{(n)},y^{(n)})$赋予一个权重$r^{(n)}$,经验风险函数为

pos [0.5997746914211707, 0.6298010406950155, 0.6466762915501375, 0.6583409563845396, 0.6672121716076366, 0.6743458028630059, 0.6802966561820678, 0.6853918153619606, 0.6898400287301549, 0.6937825086462486]



pos [0.7692838503594772, 0.7943538274241474, 0.8076013728124118, 0.816445697202922, 0.8230123767382064, 0.8281974841105273, 0.8324601230602379, 0.8360657205520801, 0.8391810046114367, 0.8419172653399102]

```python
from numpy import exp, array, random, dot,log
import numpy as np
import time
import math
class NeuralNetwork():
    def __init__(self):
        # 设置随机数种子，使每次运行生成的随机数相同
        # 便于调试
        random.seed(1)

        # 我们对单个神经元进行建模，其中有3个输入连接和1个输出连接
        # 我们把随机的权值分配给一个3x1矩阵，值在-1到1之间，均值为0。
        self.synaptic_weights_in_hide=2 * random.random((8, 8)) - 1
        self.synaptic_weights = 2 * random.random((8, 1)) - 1
        self.example_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                                 [0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
        self.training_outputs = array([[0, 1, 1, 0, 1, 1, 1, 0]])
        self.example_weight=array([[0.125],[0.125],[0.125],[0.125],[0.125],[0.125],[0.125],[0.125],[0.125]])
        # print("synaptic_weights",self.synaptic_weights)

    # Sigmoid函数, 图像为S型曲线.
    # 我们把输入的加权和通过这个函数标准化在0和1之间。
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Sigmoid函数的导函数.
    # 即使Sigmoid函数的梯度
    # 它同样可以理解为当前的权重的可信度大小
    # 梯度决定了我们对调整权重的大小，并且指明了调整的方向
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # 我们通过不断的试验和试错的过程来训练神经网络
    # 每一次都对权重进行调整
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        training_set_outputs=training_set_outputs.T
        print("test111")
        print(training_set_outputs)
        # import pdb
        # pdb.set_trace()
        for iteration in range(number_of_training_iterations):
            # 把训练集传入神经网络.
            # print("interation",training_set_inputs)
            hide_output,output = self.think(training_set_inputs)

            # 计算损失值(期望输出与实际输出之间的差。
            error = self.example_weight*(training_set_outputs - output)
            # error = training_set_outputs - output
            # print("error",error)
            www_sum=0
            for www in range(9):
                t_www=error[www][0]
                # print("www循环", www,t_www)
                self.example_weight[www][0] =math.log((1-t_www*t_www*0.5)/(t_www*t_www*0.5),math.e)
                www_sum=math.log((1-t_www*t_www*0.5)/(t_www*t_www*0.5),math.e)+www_sum

            for www in range(9):
                self.example_weight[www][0]=math.log((1-t_www*t_www*0.5)/(t_www*t_www*0.5),math.e)/www_sum
            # error = training_set_outputs - output
            sum=np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1]])
            # print(error)
            if iteration%100==0:
                print("------iteration------",iteration)
                print("Loss",np.sum(error*error*0.5))
                print("self.example_weight",self.example_weight)
            # print("training_set_outputs",training_set_outputs)
            # print("output",output)
            #     print("error",error)
            #     print("weight_error",error*self.example_weight)
                # print("self.__sigmoide_derivative",self.__sigmoid_derivative(output))
            # print("error_len",len(error))

            # 损失值乘上sigmid曲线的梯度，结果点乘输入矩阵的转置
            # 这意味着越不可信的权重值，我们会做更多的调整
            # 如果为零的话，则误区调制
            adjustment = -dot(hide_output.T, error * self.__sigmoid_derivative(output))
            adjustment_in_hide = -dot(training_set_inputs.T,
            error * self.__sigmoid_derivative(output)*self.synaptic_weights.T*self.__sigmoid_derivative(hide_output))
            # print("training_set_inputs",training_set_inputs)
            print("error",error)
            print("output",self.__sigmoid_derivative(output))
            # print("adjustment",adjustment)
            # 调制权值

            self.synaptic_weights -= adjustment
            self.synaptic_weights_in_hide-=adjustment_in_hide
            if iteration % 100 == 0:
                self.gradient_check(training_set_inputs,training_set_outputs,adjustment,adjustment_in_hide)
            # 神经网络的“思考”过程
    def think(self, inputs):
        # 把输入数据传入神经网络
        # print("think",dot(inputs,self.synaptic_weights))
        hide_output=self.__sigmoid(dot(inputs, self.synaptic_weights_in_hide))
        output=self.__sigmoid(dot(hide_output, self.synaptic_weights))
        return hide_output,output
    def gradient_check(self,training_set_inputs,training_set_outputs,adjustment,adjustment_in_hide):
        # 隐层到输出层
        gradapprox_in_to_hide=np.zeros([8,8])
        for i in range(8):
            for j in range(8):
                before = self.synaptic_weights_in_hide[i][j]
                self.synaptic_weights_in_hide[i][j] = before + 1e-7
                hide_output1, output1 = self.think(training_set_inputs)
                #计算loss
                error = training_set_outputs - output1
                loss1 = np.sum(error*error*0.5)
                self.synaptic_weights_in_hide[i][j] = before - 1e-7
                hide_output2, output2 = self.think(training_set_inputs)
                error = training_set_outputs - output2
                loss2 = np.sum(error * error * 0.5)
                #计算梯度
                gradapprox_in_to_hide[i][j] = (loss1-loss2)/(2*1e-7)
                self.synaptic_weights_in_hide[i][j] = before
        numerator = np.linalg.norm(gradapprox_in_to_hide - adjustment_in_hide)  # Step 1'
        # print("numerator",numerator)
        # print("hide_to_output",hide_to_output)
        # print("adjustment",adjustment)
        denominator = np.linalg.norm(gradapprox_in_to_hide) + np.linalg.norm(adjustment_in_hide)  # Step 2'
        difference = numerator / denominator  # Step 3'
        print("in_to_hide_difference", difference)

        hide_to_output = np.zeros([8, 1])
        for i in range(8):
            before = self.synaptic_weights[i][0]
            self.synaptic_weights[i][0] = before + 1e-7
            hide_output1, output1 = self.think(training_set_inputs)
            # 计算loss
            error = training_set_outputs - output1
            loss1 = np.sum(error * error * 0.5)
            self.synaptic_weights[i][0] = before - 1e-7
            hide_output2, output2 = self.think(training_set_inputs)
            error = training_set_outputs - output2
            loss2 = np.sum(error * error * 0.5)
            # 计算梯度
            hide_to_output[i][0] = (loss1 - loss2) / (2 * 1e-7)
            self.synaptic_weights[i][0] = before

        numerator = np.linalg.norm(hide_to_output - adjustment)  # Step 1'
        # print("numerator",numerator)
        # print("hide_to_output",hide_to_output)
        # print("adjustment",adjustment)
        denominator = np.linalg.norm(hide_to_output) + np.linalg.norm(adjustment)  # Step 2'
        difference = numerator / denominator  # Step 3'
        print("hide_to_output_difference",difference)


if __name__ == "__main__":

    # 初始化一个单神经元的神经网络
    neural_network = NeuralNetwork()

    # 输出随机初始的参数作为参照
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # 训练集共有四个样本，每个样本包括三个输入一个输出
    training_set_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                                 [0, 0, 1, 0, 1, 1, 1, 1],[0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0,1,1,1,0,0]])
    # 用训练集对神经网络进行训练
    # 迭代10000次，每次迭代对权重进行微调.
    # cir=0
    # pos=[]
    # while cir<10:
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    # 输出训练后的参数值，作为对照。
    # print("New synaptic weights after training: ")
    # print(neural_network.synaptic_weights)

    # 用新样本测试神经网络.
    print("Considering new situation [1, 0, 0,0,0,0,0,0] -> ?: ")
    hide_out, out = neural_network.think(array([1, 0, 0, 0, 0, 0, 0, 0]))
    # print(hide_out)
    print(neural_network.example_weight)
    print(out)
    # pos.append(out[0])
    # print(neural_network.synaptic_weights)
    # print(neural_network.synaptic_weights_in_hide)
    # cir = cir + 1
        # example = array(
        #     [[0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1]])
    # print("pos",pos)
```

上述代码是加上权重更新的三层网络模型。

2-3 证明略

2-4 在线性回归中，验证岭回归的解为结构风险最小化准则下的最二乘法估计，见公式(2.45)。

![1564997669673](C:\Users\xiaomin\AppData\Roaming\Typora\typora-user-images\1564997669673.png)



2-5 

在线性回归种，，若假设标签$y ∼ N (w^Tx, β)​$，并用最大似然估计来优化参数时，验证最优参数为公式(2.51)的解。

习题 **2-11** 分别用一元、二元和三元特征的词袋模型表示文本“我打了张三”和“张三打了我”，并分析不同模型的优缺点。

<u>我</u> <u>打了</u> <u>张三</u> , <u>张三</u> <u>打了</u> <u>我</u> .

一元特征：“我”，“打了”，“张三”

分别表示为：

$v_1=[1,1,1]​$ 

$v_2=[1,1,1]​$ 

二元特征： “$我”，“我打了”，“打了张三”，“张三#”，“​\$张三”, "张三打了"，“打了我”，“我#”

$v_1=[1,1,1,1,0,0,0,0]​$

$v_2=[0,0,0,0,1,1,1,1]​$ 

三元特征："$$我"，“$我打了”，“我打了张三”，“打了张三#”，“张三##”，

“$$张三”，“$张三打了”，“张三打了我”，“打了我#”，“我##”

$v_1=[1,1,1,1,1,0,0,0,0,0]$

$v_2=[0,0,0,0,0,1,1,1,1,1]​$ 



 

