#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


tf.__version__,np.__version__


# In[5]:


from tensorflow.examples.tutorials.mnist import input_data as input_data


# In[6]:


mnist = input_data.read_data_sets("/Users/dy/Documents/MNIST_data",one_hot=True)


# In[7]:


print("训练集train images shape:",mnist.train.images.shape,
      "训练集train labels shape",mnist.train.labels.shape)
print("验证集validation images shape:",mnist.validation.images.shape,
      "验证集validation labels shape",mnist.validation.labels.shape)
print("测试集test images shape:",mnist.test.images.shape,
      "测试集test labels shape",mnist.test.labels.shape)


# # 查看数据集

# In[8]:


import matplotlib.pyplot as plt

def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()


# In[9]:


mnist.train.images[0].reshape(28,28)


# In[10]:


plot_image(mnist.train.images[0])


# ## 显示标签label

# In[11]:


mnist.train.labels[0]


# ## 定义输入x和输出y的形状，理解为采用占位符进行占位
# ## None:表示样本的个数，未确定，28*28是输入的特征数目

# In[12]:


input_x = tf.placeholder(tf.float32, [None, 28*28]) / 255 # 因为每个像素取值范围是0～255
output_y = tf.placeholder(tf.int32, [None, 10])  # 10表示10个类别


# In[13]:


input_x[0]


# ## 输入层数据背reshape成4维数据，第一维是图片数量

# In[14]:


input_x_images = tf.reshape(input_x, [-1,28,28,1])
test_x = mnist.test.images[:3000] # 读取测试集图片特征，3000个
test_y = mnist.test.labels[:3000] # 读取测试集图片标签，3000个


# ## 定义卷积神经网络结构

# ## 第一层：卷积层 conv1，tf.layers.conv2d表示使用2维的卷积层

# In[20]:


conv1 = tf.layers.conv2d(inputs = input_x_images,  # 输入
                         filters = 32,  # 滤波器数量
                         kernel_size = [5, 5] , # 卷积核尺寸 
                         strides = 1, # 步长
                         padding = 'same',  # 边缘填充补0
                         activation = tf.nn.relu # 激活函数
                        )


# In[21]:


print(conv1)


# ## 第二层：池化层 pool1，最大池化

# In[29]:


pool1 = tf.layers.max_pooling2d(inputs = conv1, # 输入是上一层卷积层
                               pool_size = [2,2],# 连接池大小
                               strides = 2 # 池化步长
                               )


# In[30]:


print(pool1)


# ## 第三层：卷积层conv2，64个卷积核，每个核大小5*5

# In[31]:


conv2 = tf.layers.conv2d(inputs = pool1,  # 输入是上一层池化层
                         filters = 64,  # 滤波器数量
                         kernel_size = [5, 5] , # 卷积核尺寸 
                         strides = 1, # 步长
                         padding = 'same',  # 边缘填充补0
                         activation = tf.nn.relu # 激活函数
                        )


# In[32]:


print(conv2)


# ## 第四层：池化层pool2，最大池化

# In[33]:


pool2 = tf.layers.max_pooling2d(inputs = conv2, 
                               pool_size = [2,2],# 连接池大小
                               strides = 2 # 池化步长
                               )


# In[34]:


print(pool2)


# ## 将两次卷积两次池化的结果降成一维7x7x64=3136

# In[35]:


flat = tf.reshape(pool2,[-1,7*7*64])


# ## 第五层：全联接层dense inputs：输入，units：输出的大小

# In[37]:


dense = tf.layers.dense(inputs = flat,
                        units = 1024, 
                        activation = tf.nn.relu
                       )


# In[38]:


print(dense)


# ## dropout操作，丢弃率rate=0.5，即一半的神经元丢弃不工作

# In[40]:


dropout = tf.layers.dropout(inputs = dense,
                            rate = 0.5)


# In[42]:


print(dropout)


# ## 输出层

# In[43]:


outputs = tf.layers.dense(inputs = dropout, 
                          units = 10)


# In[44]:


print(outputs)


# # 为反向传播作准备

# ## 计算损失，使用交叉熵损失函数

# In[105]:


loss = tf.losses.softmax_cross_entropy(onehot_labels = output_y, #真实结果
                                       logits = outputs # 预测结果
                                      )
print(loss)


# ## 定义训练操作

# In[106]:


train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss) # 学习率为0.001
print(train_op)


# ## 定义模型的性能评价指标：准确率

# In[107]:


accuracy_op = tf.metrics.accuracy(labels = tf.argmax(output_y,axis = 1), # 返回张量维度上的最大值的索引
                                  predictions = tf.argmax(outputs, axis = 1) #返回张量维度上的最大值的索引
                                 )
print(accuracy_op) #打印这个张量


# # 准备工作结束，正式开始训练

# In[116]:


# 初始化所有的变量
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
# 进行1000次迭代
for i in range(1000):
    batch = mnist.train.next_batch(50)
    # batch的50副图片的特征batch[0]和标签batch[1]
    train_loss = sess.run(loss, {input_x:batch[0], output_y:batch[1]})
    train_op_ = sess.run(train_op, {input_x:batch[0], output_y:batch[1]})
    # 每迭代100次就输出训练的损失和测试集（3000个）的准确率
    if i%100 ==0 :
        # 计算测试准确率
        test_accuracy = sess.run(accuracy_op, {input_x:test_x, output_y:test_y} )
        print("test_accuracy:",test_accuracy)
        print("Step=%d, Train loss = %.4f, Test accuracy = %.2f"%(i,train_loss, test_accuracy[0]))


# # 模型训练完毕，检测一下预测效果

# In[120]:


# 预测结果：test_output
test_output = sess.run(outputs, {input_x:test_x[0:20]})
# np.argmax(a, axis=None, out=None)：返回沿轴axis最大值的索引
# a ：输入一个array类型的数组。
# axis：参数为None时默认比较整个数组，参数为0按列比较，参数为1按行比较。
inferenced_y = np.argmax(test_output, 1)
# 真实的标签
real_y = np.argmax(test_y[0:20],1)
print(inferenced_y, 'inferenced numbers（预测的标签）')
print(real_y,'real numbers（真实的标签）')


# ## 关闭session

# In[121]:


#sess.close()

