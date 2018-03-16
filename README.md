# TensorNewtonRaphson (TNR)
This package implements the numerical method for root finding, Newton-Raphson,
 using the TensorFlow library. 

## Introduction
The Newton-Raphson is a method for numerically finding the roots of any
given equation, provided the derivative exists, is known and the method succeeds in
converging to the answer.

![TNR formula](images/newton_raphson.png)

By making use of TensorFlow, this code allows the method to be run without
having to calculate the derivative and also make use of GPU power through
CUDA if necessary.

## Executing
To run the algorithm, all you have to do is configure the desired equation,
initial guess and precision on the __main__.py file and then execute the following in your
command shell:

```shell
$ python __main__.py
```

Or, if you prefer configuring it in the command shell, you can also execute
the algorithm like this:

```shell
$ python __main__.py --initial_guess=1.0 --precision=1e-9 --equation="tf.pow(x, 2)"
```