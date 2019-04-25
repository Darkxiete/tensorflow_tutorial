from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import typing
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Preprocess the data
# train_images = train_images / 255.0
# train_labels = train_labels / 255.0
train_images = train_images / 255.0

test_images = test_images / 255.0


"""
Build the model
第一层扁平化，每个28 * 28 的图片都变成1 x 784 的一维向量
第二层全联接，128个神经元
第三层全联接，10个神经元，对应10个输出
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
Compile the model
定义损失函数，然后使用优化方法来优化损失函数，最后选择评价指标
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
Train the model 
"""
model.fit(train_images, train_labels, epochs=5)

"""
Evaluate accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)