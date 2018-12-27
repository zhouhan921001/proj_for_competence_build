import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os

import load_data as ld

##################################
# define constant variable       #
##################################
image_size = 32
num_channels = 3
filter_size = 3

filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

fc_size = 128             # Number of neurons in fully-connected layer.

batch_size = 64           # Size of samples for once

test_batch_size = 256

os.environ['CUDA_VISIBLE_DEVICES'] = '0'       # GPU 0 is visible

##################################
#          load dataSet          #
##################################
path_train = "/home/han/project/dataSet/cifar-10-batches-py/data_batch_"
data_train, cls_train = ld.merge_train_data(path_train, 10, 5)

path_test = "/home/han/project/dataSet/cifar-10-batches-py/test_batch"
data_test, cls_test = ld.load_data(path_test)

##################################
#       define placeHolder       #
##################################
x = tf.placeholder(shape=[None, image_size, image_size, num_channels], dtype=tf.float32, name='x')
y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y')    # can't use -1, must use None
y_cls = tf.argmax(y, axis=1)


##################################
#          define model          #
##################################
def conv_layer(input_data, layer_num, num_filters, filter_size, filter_channels, pooling):

    with tf.variable_scope('conv_layer'+str(layer_num)) as scope:
        weight = tf.get_variable(name='weight',
                                 shape=[filter_size, filter_size, filter_channels, num_filters], dtype=tf.float32)
        bias = tf.get_variable(name='bias', shape=[num_filters], dtype=tf.float32)

    output = tf.nn.conv2d(input=input_data, filter=weight, strides=[1, 2, 2, 1], padding='SAME')

    output += bias

    if pooling:
        output = tf.nn.max_pool(value=output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    output = tf.nn.relu(output)

    return output


def flatten_layer(input_data):

    input_shape = input_data.get_shape()
    num_elements = input_shape[1]*input_shape[2]*input_shape[3]
    input_data = tf.reshape(input_data, [-1, num_elements])

    return input_data, num_elements


def fc_layer(input_data, layer_num, num_input_data,num_output_data, use_relu=True):

    with tf.variable_scope('fc_layer'+str(layer_num)):
        weight = tf.get_variable(name='weight', shape=[num_input_data, num_output_data])
        bias = tf.get_variable(name='bias', shape=[num_output_data])

    output = tf.matmul(input_data, weight)

    output += bias

    if use_relu:
        output = tf.nn.relu(output)

    return output


############################################
#      construct computational graph       #
############################################
conv1_output = conv_layer(x, 1, num_filters1, filter_size1, 3, True)

conv2_output = conv_layer(conv1_output, 2, num_filters2, filter_size2, num_filters1, True)

flatten_output, num_layer2 = flatten_layer(conv2_output)

fc1_output = fc_layer(flatten_output, 1, num_layer2, fc_size)

fc2_output = fc_layer(fc1_output, 2, fc_size, 10)

y_pred = tf.nn.softmax(fc2_output)

y_pred_true = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2_output, labels=y)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

num_same = tf.equal(y_pred_true, y_cls)
print(y_pred_true.shape)
print(y_cls.shape)

accuracy = tf.reduce_mean(tf.cast(num_same, tf.float32))


############################################
#           create and run session         #
############################################
session = tf.Session()

session.run(tf.global_variables_initializer())

############################################
#              create a saver              #
############################################
saver = tf.train.Saver()

save_dir = 'checkpoint/simple_conv_net_cifar10/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir,'simple_conv_net_cifar10')


def optimize(input_x, input_y, iter_num):

    """
    iter_batch_head = 0
    x_len = len(input_x)
    print(x_len)
    iter_num = (x_len // batch_size) + 1
    print(iter_num)
    """
    start_time = time.time()

    with tf.device('/device:GPU:0'):
        for i in range(iter_num):

            x_batch, y_batch = ld.random_batch(batch_size, input_x, input_y)
            """
            if iter_batch_head+batch_size > x_len:
                iter_batch_end = x_len
            else:
                iter_batch_end = iter_batch_head+batch_size
            x_batch = input_x[iter_batch_head: iter_batch_end, :]
            y_batch = input_y[iter_batch_head: iter_batch_end]
            """

            feed = {x:x_batch, y:y_batch}

            session.run(optimizer, feed_dict=feed)

            if i % 100 == 0:

                cost_batch = session.run(cost,feed_dict=feed)
                accuracy_batch = session.run(accuracy,feed_dict=feed)
                msg = "cost : {}, accuracy : {}, iter num : {}\n"
                print(msg.format(cost_batch, accuracy_batch, i))
                saver.save(session, save_path=save_path)

        # iter_batch_head = iter_batch_end

    end_time = time.time()

    total_time = end_time - start_time

    print("The total of time is {}".format(total_time))


def predict_cls(test_x, test_y):

    num_test = len(test_x)
    saver.restore(session, save_path)
    cls_test = np.zeros(shape=num_test,dtype=int)
    cls_test_true = np.zeros(shape=[num_test, 10],dtype=int)
    print(test_y.shape)

    for j in range(num_test):
        cls_test_true[j, test_y[j]] = 1

    i = 0
    while i < num_test:
        j = min(i+test_batch_size, num_test)
        test_dict = {x: test_x[i:j, :],
                     y: cls_test_true[i:j, :]}
        cls_test[i:j] = session.run(y_pred_true, feed_dict=test_dict)

        i += test_batch_size

    correct_test = tf.equal(cls_test, test_y)
    pre_accuracy = tf.reduce_mean(tf.cast(correct_test, dtype=tf.float32))
    accu = session.run(pre_accuracy, feed_dict=test_dict)
    print(pre_accuracy.shape)
    print("predict accuracy : {}\n".format(accu))


if __name__ == '__main__':

    optimize(data_train, cls_train, 500000)
    predict_cls(data_test, cls_test)
