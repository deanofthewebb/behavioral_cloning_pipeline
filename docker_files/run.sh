#!/bin/bash
set -e
. activate tf-gpu
exec time python3 -c "import tensorflow as tf;\
                              hello = tf.constant('Hello, TensorFlow!');\
                              sess = tf.Session();\
                              print(sess.run(hello));\
                              print(tf.__version__);\
                              import os;\
                              print(os.path.dirname(tf.__file__))"
if [ -z "$1" ]
  then
    jupyter notebook
elif [ "$1" == *".ipynb"* ]
  then
    jupyter notebook "$1"
else
    exec "$@"
fi
