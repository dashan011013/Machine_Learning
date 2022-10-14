import keras
from keras import backend as K
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128  # 一次训练所选取的样本数
num_classes = 10  # 分类个数
epochs = 10  # 训练轮数

# 读取已下载到本地的数据集
f = np.load('C:/Users/Administrator/.keras/datasets/mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
# print(x_train.shape, y_train.shape)


# 数据预处理
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')  # 转换数据类型
x_test = x_test.astype('float32')
x_train /= 255  # 归一化
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)  # 将整形数组转化为二元类型矩阵
y_test = keras.utils.to_categorical(y_test, num_classes)
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')


# 创建CNN模型
model = Sequential()  # 这里采用顺序模型构建CNN
# 输入层，这里指定输入数据形状为28*28*1 卷积核数量为32 形状为3*3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# 添加中间层
model.add(Conv2D(64, (3, 3), activation='relu'))  # 卷积层
model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化层
model.add(Dropout(0.25))  # 通过Dropout防止过拟合
model.add(Flatten())  # 展平层
model.add(Dense(256, activation='relu'))  # 全连接层
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# 损失函数
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# 训练模型
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
print("模型训练完成")

# 模型评估
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss: ', score[0])
print('test accuracy: ', score[1])
