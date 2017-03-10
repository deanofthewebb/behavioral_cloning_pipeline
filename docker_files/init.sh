#!/bin/bash
# apt upgrade
# apt install aptitude
# aptitude    update
# aptitude -y upgrade
# aptitude install linux-image-extra-`uname -r`
# sh -c "wget -qO- https://get.docker.io/gpg | apt-key add -"
# sh -c "echo deb http://get.docker.io/ubuntu docker main\
# > /etc/apt/sources.list.d/docker.list"
# aptitude    update
# sudo aptitude install lxc-docker

apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get update
apt-get install docker-ce

python3 /home/ubuntu/sharefolder/behavioral_cloning/docker_files/docker_setup.py
pip install docker
python3 /home/ubuntu/sharefolder/behavioral_cloning/docker_files/client_script.py

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
