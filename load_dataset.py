import random
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image,a,h,j):
    plt.imshow(image.reshape(a,a),cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('image/layer{}_{}.png'.format(h, j))
    # plt.show()

def normlize_image(image):
    min_value = min(image)
    max_value = max(image)
    for l in range(len(image)):
        value =255* (image[l] - min_value) / (max_value - min_value)
        image[l] = value
    return image
# 加载并处理数据
# 数据转换
# tr_d是由50000个长度为784的numpy.ndarray组成的tuple
# 转换后的training_inputs是由50000个长度为784的numpy.ndarray组成的list
def load_data(filepath):
    # 读取数据
    f = gzip.open(filepath, 'rb')
    tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
    f.close()

    training_inputs = np.empty([len(tr_d[0]), 784])
    training_results =np.empty([len(tr_d[1]), 10])
    # 训练集
    for i in range(len(tr_d[0])):
        x = np.reshape(tr_d[0][i], (1, 784))
        training_inputs[i] = x

    for i in range(len(tr_d[1])):
        y = one_hot(tr_d[1][i])
        training_results[i] = y

    # 测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    # return (training_data, test_data)
    x = np.array(training_inputs)
    y = np.array(training_results)
    return (x, y)

# 独热编码
def one_hot(j):
    e = np.zeros((1, 10))
    e[0][j] = 1.0
    return e


def txt_to_numpy(filename, row, colu):
    """
	filename:是txt文件路径
	row:txt文件中数组的行数
	colu:txt文件中数组的列数
	"""
    file = open (filename)
    lines = file.readlines()
    # print(lines)
	# 初始化datamat
    datamat = np.zeros((row, colu))

    row_count = 0

    for line in lines:
        # 写入datamat
        line = line.strip().split(' ')
        datamat[row_count,:] = line[:]
        row_count += 1


    return datamat
