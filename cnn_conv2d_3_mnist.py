#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:10:07 2019

@author: zixks
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


def wb_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def conv2d_kernel(shape=[3, 3, 1, 1]):
    kernel = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(kernel)


def input_layer(x, name='input_layer'):
    with tf.name_scope(name):
        img = tf.reshape(x, [-1, 28, 28, 1])
    return img


def max_pool_2x2(x, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1], padding = 'SAME')


def conv2d_layer(x, kernel, bias, name, activation=None, strides=(1, 1, 1, 1), 
                 padding='SAME'):
    
    if name.strip() == '':
        raise Exception('name can not be null')
        
    if kernel is None or x is None:
        raise Exception('x and kernel can not be null')
    
    with tf.name_scope(name):
        conv2dout = tf.nn.conv2d(input=x, filter=kernel, strides=strides, 
                              padding=padding)
        layer = conv2dout + bias
        
        if activation is not None:
            return tf.nn.relu(layer)
        else:
            return layer

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    in_layer = input_layer(x, 'input_layer')
    
    #########################################################################
    conv_1_1 = conv2d_layer(x=in_layer, kernel=wb_variable([3, 3, 1, 32]), 
                          bias=wb_variable([32]), activation=tf.nn.relu, 
                          name='conv_1_1')
    conv_1_2 = conv2d_layer(x=conv_1_1, kernel=wb_variable([3, 3, 32, 32]),
                          bias=wb_variable([32]), activation=tf.nn.relu,
                          name='conv_1_2')
    pool_1 = max_pool_2x2(x=conv_1_2, name='pool_2')
    
    # 14
    #########################################################################
    conv_2_1 = conv2d_layer(x=pool_1, kernel=wb_variable([3, 3, 32, 64]), 
                          bias=wb_variable([64]), activation=tf.nn.relu, 
                          name='conv_2_1')
    conv_2_2 = conv2d_layer(x=conv_2_1, kernel=wb_variable([3, 3, 64, 64]), 
                          bias=wb_variable([64]), activation=tf.nn.relu, 
                          name='conv_2_2')
    pool_2 = max_pool_2x2(x=conv_2_2, name='pool_2')
    
    # 7
    #########################################################################
    conv_3_1 = conv2d_layer(x=pool_2, kernel=wb_variable([3, 3, 64, 128]), 
                          bias=wb_variable([128]), activation=tf.nn.relu, 
                          name='conv_3_1')
    conv_3_2 = conv2d_layer(x=conv_3_1, kernel=wb_variable([3, 3, 128, 128]), 
                          bias=wb_variable([128]), activation=tf.nn.relu, 
                          name='conv_3_2')
    
    
    
    with tf.name_scope('FULL_CONNECT_1'):
        w6 = wb_variable([128 * 7 * 7, 1024])
        b6 = wb_variable([1024])
        f_6_in = tf.reshape(conv_3_2, shape=[-1, 128 * 7 * 7])
        f_6_out = tf.nn.relu(tf.matmul(f_6_in, w6 ))
    

    with tf.name_scope('DROPOUT_1'):
        keep_prob = tf.placeholder("float")
        dropout_1 = tf.nn.dropout(f_6_out, keep_prob=keep_prob)
    
    
    with tf.name_scope('SOFTMAX_1'):
        W_fc2 = wb_variable([1024, 10])
        b_fc2 = wb_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(dropout_1, W_fc2) + b_fc2)
    
    
    loss = -tf.reduce_sum(y * tf.log(y_conv)) #计算交叉熵
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        tf.summary.FileWriter("logs/", sess.graph)
        for i in range(100000):
            batch = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1], 
                                            keep_prob:0.5})
            
            if i % 100 == 0:
                lt = loss.eval(session=sess, feed_dict = {x:batch[0], y:batch[1], keep_prob:1.0})
                ld = loss.eval(session=sess, feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
                train_accuracy = accuracy.eval(session = sess,
                                               feed_dict = {x:batch[0], y:batch[1], keep_prob:1.0})
                test = accuracy.eval(session = sess,
                                     feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
                print("epoch {}, train accuracy {}%, dev accuracy {}%, train loss {}%, dev loss {}%".format(
                        i, round(train_accuracy * 100, 2), round(test * 100, 2), round(lt * 100, 2), round(ld * 100, 2)))
