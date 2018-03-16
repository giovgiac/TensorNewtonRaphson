# Copyright 2018 Giovanni Giacomo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Newton-Raphson parameters
tf.flags.DEFINE_float("initial_guess", 1.0, "The starting guess to plug-in the Newton-Raphson method.")
tf.flags.DEFINE_float("precision", 1e-12, "The precision to achieve before stopping the method.")
tf.flags.DEFINE_string("equation", "2 * tf.cos(3 * x) - tf.exp(x)", "The equation to evaluate the root of.")


def f(x):
    """Evaluates the function at the given point and calculates it's derivative, returning their division.

    ...

    :param x: The point at which to calculate the function's and it's derivative's values.

    :return: The division of the value of the function by the value of it's derivative.
    """
    fx = eval(FLAGS.equation)
    dx = tf.gradients(fx, x)

    return fx / dx


def body(x, x_0, p, k):
    """The body of the while loop that runs the Newton-Raphson algorithm until the desired precision is achieved.

    ...

    :param x: The current point for which the Newton-Raphson was calculated.
    :param x_0: The previous point for which the Newton-Raphson was calculated.
    :param p: The precision until which to run the method.
    :param k: The number of steps taken so far.

    :return: The list formed by [new_point, old_point, precision, steps].
    """
    x_0 = tf.identity(x)
    x = tf.reshape(x_0 - f(x_0), [])
    k = tf.add(k, 1)

    return [x, x_0, p, k]


def condition(x, x_0, p, k):
    """The condition of the while loop that runs the Newton-Raphson algorithm until the desired precision is achieved.

    ...

    :param x: The current point for which the Newton-Raphson was calculated.
    :param x_0: The previous point for which the Newton-Raphson was calculated.
    :param p: The precision until which to run the method.
    :param k: The number of steps taken so far.

    :return: True if the precision has been achieved, False otherwise.
    """
    return tf.reshape(tf.abs(x - x_0), []) > p


def newton(x_0, p):
    """The Newton-Raphson algorithm implemented using TensorFlow routines for maximum efficiency and practicality.

    ...

    :param x_0: The initial guess to use when running the Newton-Raphson for the first time.
    :param p: The precision until which to run the method.
    """
    with tf.Session() as sess:
        p = tf.constant(
            name="precision",
            shape=[],
            dtype=tf.float32,
            value=p)

        k = tf.Variable(
            name="global_step",
            expected_shape=[],
            dtype=tf.int64,
            initial_value=0,
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        x = tf.Variable(
            name="root",
            expected_shape=[],
            dtype=tf.float32,
            initial_value=tf.reshape(tf.constant(x_0) - f(tf.constant(x_0)), []),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        x_0 = tf.Variable(
            name="initial_guess",
            expected_shape=[],
            dtype=tf.float32,
            initial_value=x_0,
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        loop = tf.while_loop(condition, body,
                             loop_vars=[x, x_0, p, k])

        # Initialize variables
        sess.run(init)

        # Output results
        result = sess.run(loop)
        tf.logging.info("RESULT: root %f found in %d steps.\n",
                        result[0],
                        result[-1])


def main(unused_argv):
    tf.logging.set_verbosity(3)
    newton(FLAGS.initial_guess, FLAGS.precision)


if __name__ == '__main__':
    tf.app.run()
