# coding=utf8
import tensorflow as tf
from tensorflow import keras
# from tensorflow import examples
from tensorflow.examples.tutorials.mnist import input_data

"""
参考这个代码
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
INPUT_NODE = 784
OUTPUT_NODE = 10

# 只有一层隐含层
LAYER1_NODE = 500

# 一个batch中的样例数
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
REGULARIZATION_RATE
TRAINING_STEPS = 30000
# TODO 这个东西到底有什么用？
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights_1, biases_1, weights_2, biases_2):
    # 不使用滑动平均
    if avg_class:
        # 这里是两个参数都要滑动平均，之前打错了导致计算准确率的时候出错
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights_1)) + avg_class.average(biases_1))
        return tf.matmul(layer_1, avg_class.average(weights_2)) + avg_class.average(biases_2)
    else:
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights_1) + biases_1)
        return tf.matmul(layer_1, weights_2) + biases_2


def train(mnist):
    # shape = (500, 784)
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x_input")
    # shape = (500, 10)
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_input")

    # 不在两个标准差之内的随机数会被重新选择
    weights_1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases_1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights_2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases_2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 预测值, shape = (500 ,10)
    y = inference(x, None, weights_1, biases_1, weights_2, biases_2)

    # 滑动平均
    # TODO trainable=False 似乎是不求梯度的意思，如果是True的话就会放到一个集合里，那个集合里都会对变量求梯度
    global_step = tf.Variable(0, trainable=False)
    # 有什么意义
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    average_y = inference(x, variables_averages, weights_1, biases_1, weights_2, biases_2)

    # 交叉熵cross_entropy
    # DONE y 和 y_ 哪个该用tf.argmax?
    # label shape应该是 (BATCH_SIZE, ), logits shape 应该是 (BATCH_SIZE, NUM_CLASS)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    # 一个batch里的交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights_1) + regularizer(weights_2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # TODO 什么用？
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 之前写成了train_step, 导致准确率计算错误
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps(s), test accuracy "
              "using average model is %g " % (TRAINING_STEPS, test_acc))


def main():
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    mnist = input_data.read_data_sets("./data", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    print(tf.__version__)
    # TODO 准确率算错了，第1000步是0.093而不是0.93，现在是错误的
    # 滑动平均的地方写错了
    main()
