#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:18:59 2019
MNIST ResNet
@author: Palour
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# generate weight param
def weight_variable(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)


x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
y = tf.placeholder(dtype=tf.float32, shape=(None, 10))

img = tf.reshape(tensor=x, shape=(-1, 28, 28, 1))

conv_1_1 = tf.nn.relu(
        tf.nn.conv2d(input=img, filter=weight_variable([5, 5, 1, 1]), 
                     strides=(1,1,1,1), padding='SAME') + bias_variable([1]))

conv_1_2 = tf.nn.relu(
        tf.nn.conv2d(input=img, filter=weight_variable([5, 5, 1, 1]),
                     strides=(1,1,1,1), padding='SAME') + bias_variable([1]))


res_2_1 = conv_1_2 + img

conv_3_1 = tf.nn.relu(
        tf.nn.conv2d(input=res_2_1, filter=weight_variable([5, 5, 1, 32]),
                     strides=(1,1,1,1), padding='SAME') + bias_variable([32]))

pool_3_1 = tf.nn.max_pool(value=conv_3_1, ksize=(1, 2, 2, 1), 
                          strides=(1, 2, 2, 1), padding='SAME')

# 14 14 32

fc_4_1 = tf.reshape(pool_3_1, shape=(-1, 14 * 14 * 32))

fc_4_2 = tf.nn.relu(
        tf.add(tf.matmul(fc_4_1, weight_variable([14 * 14 * 32, 1024])), 
               bias_variable([1024]))
        )
        
keep_prob = tf.placeholder("float")
dropout_5_1 = tf.nn.dropout(fc_4_2, keep_prob=keep_prob)

softmax_6_1 = tf.nn.softmax(tf.matmul(dropout_5_1, weight_variable([1024, 10])) + bias_variable([10]))

loss = -tf.reduce_sum(y * tf.log(softmax_6_1)) #计算交叉熵
    
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(softmax_6_1,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
    
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    tf.summary.FileWriter("logs/", sess.graph)
    last = mnist.train.epochs_completed
    if mnist.train.epochs_completed < 10000:
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], 
                                        keep_prob:0.5})
        
        if last != mnist.train.epochs_completed:
            last = mnist.train.epochs_completed
            lt = loss.eval(session=sess, feed_dict = {x:batch[0], y:batch[1], keep_prob:1.0})
            ld = loss.eval(session=sess, feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
            train_accuracy = accuracy.eval(session = sess,
                                           feed_dict = {x:batch[0], y:batch[1], keep_prob:1.0})
            test = accuracy.eval(session = sess,
                                 feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
            print("epoch {}, train accuracy {}%, dev accuracy {}%".format(
                    mnist.train.epochs_completed, round(train_accuracy * 100, 2), round(test * 100, 2)))


