

## Use something a for loop like this to generate and run the various Hyperparameters ##
for CAPABILITY in $COMPUTE_CAPABILITIES; do
  if [[ ! "$CAPABILITY" =~ [0-9]+.[0-9]+ ]]; then
    echo "Invalid compute capability: " $CAPABILITY
    ALL_VALID=0
    break
  fi
done

## Use the above to generate commandline-arguments like so ##




## From Dockerfile_TF_Nvidia.GPU











# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        vim \
        curl \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        openjdk-8-jdk \
        build-essential \
        swig \
	python \
        python-numpy \
        python3-numpy \
        python-dev \
        python3-dev \
        python-pip \
        python3-pip \
        python-virtualenv \
        python-wheel \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
	tcl-dev \
	tk-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh tmp/Miniconda3-4.2.12-Linux-x86_64.sh
RUN bash tmp/Miniconda3-4.2.12-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/

COPY environment-gpu.yml  ./environment.yml
RUN conda env create -f=environment.yml --name carnd-term1 --debug -v -v

# cleanup tarballs and downloaded package files
RUN conda clean -tp -y

# Set up our notebook config.
COPY docker_files/jupyter_notebook_config.py /root/.jupyter/

# Term 1 workdir
RUN mkdir /src
WORKDIR "/src"

# Make sure CUDNN is detected Symlink (Using cuda-8.0 binaries)
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda-8.0/lib64/libcudnn.so.5 /usr/local/cuda-8.0/lib64/libcudnn.so

#Copy - I think there's a better way
COPY  /usr/local/cuda-8.0/ /usr/local/cuda-8.0/

# Install Tensorflow GPU 1.0 and test
RUN pip3 install --upgrade pip && \
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp35-cp35m-linux_x86_64.whl
RUN nvidia-docker-plugin run -l 2222 --rm tensorflow/tensorflow:latest-gpu python -c 'import tensorflow as tf ; print tf.__version__'
RUN time python3 -m tensorflow.models.image.mnist.convolutional

# TensorBoard
EXPOSE 6006
# Jupyter
EXPOSE 5000-10000
# Flask Server
EXPOSE 4567

## Two Birds, One Stone
# 1. sources conda environment
# 2. prevents the zombie container issue when started as pid 1
COPY docker_files/run.sh /

RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]
