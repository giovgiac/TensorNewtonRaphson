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

import main


class NewtonTest(tf.test.TestCase):
    """This unittest tests the functionality of the root-finding method in the main module.

    ...

    """

    def test(self):
        main.FLAGS.equation = "2 * tf.cos(3 * x) - tf.exp(x)"
        main.FLAGS.initial_guess = 1.0
        main.FLAGS.precision = 1e-6

        res = main.newton(main.FLAGS.initial_guess, main.FLAGS.precision)
        self.assertGreaterEqual(abs(res[0] - 0.410966), main.FLAGS.precision)


class NewtonDivTest(tf.test.TestCase):
    """This unittest tests the functionality of the function by derivative division function in the main module.

    ...

    """

    def test(self):
        main.FLAGS.equation = "tf.pow(x, 2) - 9"
        main.FLAGS.initial_guess = 2.0
        main.FLAGS.precision = 1e-9

        div = main.f(tf.constant(main.FLAGS.initial_guess))
        with self.test_session():
            div = div.eval()

        self.assertEqual(div, -1.25)


if __name__ == '__main__':
    tf.test.main()
