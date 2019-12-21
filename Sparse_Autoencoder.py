#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

N_INPUT = 4
N_HIDDEN = 100
N_OUTPUT = N_INPUT
BETA = tf.constant(3.0)
LAMBDA = tf.constant(.0001)
EPSILON = .00001
RHO = .1


def diff(input_data, output_data):
    ans = tf.reduce_sum(tf.pow(tf.subtract(output_data, input_data), 2))
    return ans


def main(_):
    weights = {
        'hidden': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN]), name="w_hidden"),
        'out': tf.Variable(tf.random_normal([N_HIDDEN, N_OUTPUT]), name="w_out")
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN]), name="b_hidden"),
        'out': tf.Variable(tf.random_normal([N_OUTPUT]), name="b_out")
    }

    def KLD(p, q):
        invrho = tf.subtract(tf.constant(1.), p)
        invrhohat = tf.subtract(tf.constant(1.), q)
        addrho = tf.add(tf.multiply(p, tf.log(tf.div(p, q))), tf.multiply(invrho, tf.log(tf.div(invrho, invrhohat))))
        return tf.reduce_sum(addrho)

    with tf.name_scope('input'):
        # input placeholders
        x = tf.placeholder("float", [None, N_INPUT], name="x_input")
        # hidden = tf.placeholder("float", [None, N_HIDDEN], name = "hidden_activation")

    with tf.name_scope("hidden_layer"):
        # from input layer to hidden layer
        hiddenlayer = tf.sigmoid(tf.add(tf.matmul(x, weights['hidden']), biases['hidden']))

    with tf.name_scope("output_layer"):
        # from hidden layer to output layer
        out = tf.nn.softmax(tf.add(tf.matmul(hiddenlayer, weights['out']), biases['out']))

    with tf.name_scope("loss"):
        # loss items
        cost_J = tf.reduce_sum(tf.pow(tf.subtract(out, x), 2))

    with tf.name_scope("cost_sparse"):
        # KL Divergence items
        rho_hat = tf.div(tf.reduce_sum(hiddenlayer), N_HIDDEN)
        cost_sparse = tf.multiply(BETA, KLD(RHO, rho_hat))

    with tf.name_scope("cost_reg"):
        # Regular items
        cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(weights['hidden']), tf.nn.l2_loss(weights['out'])))

    with tf.name_scope("cost"):
        # cost function
        cost = tf.add(tf.add(cost_J, cost_reg), cost_sparse)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        init = tf.initialize_all_variables()
        sess.run(init)

        input_data = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], float)

        for i in range(10000):
            sess.run(optimizer, feed_dict={x: input_data})
            if i % 100 == 0:
                tmp = sess.run(out, feed_dict={x: input_data})
                print("i=",i, "out=",sess.run(diff(tmp, input_data)))

        tmp = sess.run(out, feed_dict={x: input_data})
        print("tmp:",tmp)


if __name__ == '__main__':
    tf.app.run()
