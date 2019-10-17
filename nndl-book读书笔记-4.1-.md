nndl-book读书笔记-4.1-

##### 1. tensorflow实现三层前馈神经网络：

```python
import tensorflow as tf
# 定义变量
w1 = tf.Variable(tf.random_normal([2,3],stddev=1),name="w1")
w2 = tf.Variable(tf.random_normal([3,1],stddev=1),name="w2")
biases1 = tf.Variable(tf.zeros([3]),name="b1")   # 隐藏层的偏向bias    [ 0. 0. 0.]
biases2 = tf.Variable(tf.zeros([1]),name="b1")   # 输出层的偏向bias   [0.]
x = tf.constant([[0.7,0.9]])
# 定义前向传播
a = tf.matmul(x,w1) + biases1
y = tf.matmul(a,w2) + biases2
# 调用会话函数输出
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
```

##### 2. tensorflow三层反向梯度传播神经网络：

```python
import tensorflow as tf
import numpy as np

#创建一个神经网络层
def add_layer(input,in_size,out_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]) +0.1 )
    W_mul_x_plus_b=tf.matmul(input,Weight) + biases
    #根据是否有激活函数
    if activation_function == None:
        output=W_mul_x_plus_b
    else:
        output=activation_function(W_mul_x_plus_b)
    return output

x_data = [[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
                              [0,0,1,0,1,1,1,1],[0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]]
y_data = [[0], [1], [1], [0],[1] ,[1], [1], [0], [0]]
x_test= [[0, 0, 1,0,0,0,0,0]]
y_test = [1]
xs=tf.placeholder(tf.float32,[None,8])
ys=tf.placeholder(tf.float32,[None,1])
#定义一个隐藏层
hidden_layer1=add_layer(xs,8,8,activation_function=tf.nn.sigmoid)
#定义一个输出层
prediction=add_layer(hidden_layer1,8,1,activation_function= tf.nn.sigmoid)
# 求解神经网络参数
# 1.定义损失函数
loss = tf.reduce_sum(tf.square(ys - prediction),1)
# 2.定义训练过程
train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)  # 梯度下降法使误差最小，学习率为0.1

init = tf.global_variables_initializer()  # 变量初始化
sess = tf.Session()
sess.run(init)  # 执行初始化
# 3.进行训练
for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # 训练
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))  # 输出当前loss
print(sess.run(prediction,feed_dict={xs: x_test}))

# 关闭sess
sess.close()

```

