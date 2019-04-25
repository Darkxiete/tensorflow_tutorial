import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

# 只有一层隐含层
LAYER_NODE = 500

# 一个batch中的样例数
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
# TODO 这个东西到底有什么用？
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights_1, biases_1, weights_2, biases_2):
    # 不适用滑动平均
    if avg_class:
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights_1)) + biases_1)
        return tf.matmul(layer_1, avg_class.average(weights_2) + biases_2)
    else:
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights_1) + biases_1)
        return tf.matmul(layer_1, weights_2) + biases_2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_input")

    # 不在两个标准差之内的随机数会被重新选择
    weights_1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    biases_1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))

    weights_2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    biases_2 = tf.Variable(tf.constant(0.1, shape=OUTPUT_NODE))

    # 预测值
    y = inference(x, None, weights_1, biases_1, weights_2, biases_2)

    # 滑动平均
    # TODO trainable=False 似乎是不求梯度的意思，如果是True的话就会放到一个集合里，那个集合里都会对变量求梯度
    global_step = tf.constant(0, trainable=False)
    # 有什么意义
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables)

    average_y = inference(x, variable_average, weights_1, biases_1, weights_2, biases_2)

    # 交叉熵cross_entropy
    # TODO y 和 y_ 哪个该用tf.argmax?
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    # 一个batch里的交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights_1) + regularizer(weights_2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)


train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step, varialbes_average_op]):
