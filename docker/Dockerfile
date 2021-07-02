FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# How to run?

# docker run -it --rm -c 4 -m 25g --name $USER-discovery-challenge -u $USER -e "constraint:s876node==s876gn01" -v /rdata/s01b_ls8_000/datensaetze/ariel-space-mission:/data/ --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidiactl --device /dev/nvidia-uvm s876cnsm:5000/heppe/discovery:1 /bin/bash

# Add users and groups for NFS.
RUN groupadd --gid 9001 s876clal && \
    groupadd --gid 9002 s876ls8  && \
    groupadd --gid 9019 s876z    && \
    useradd --gid 9002 -G s876clal,s876z -m --uid 8030 heppe && \
    useradd --gid 9002 -G s876clal -m --uid 8230 bunse

# Install some software
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install ssh git build-essential \
        nano        \
        vim         \
        less        \
        python3     \
        python3-pip \
        gdb         \
        htop        \
        cmake       \
        wget        \
        gdb         \
        libnvinfer6 \
        libnvinfer-plugin6 \
    && rm -rf /var/lib/apt/lists/*

# Add nvidia driver
COPY nv-driver-libs.418.43 /usr/local/s876-nv-driver
COPY s876-nv-driver.conf /etc/ld.so.conf.d/

ENV LD_LIBRARY_PATH /usr/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/include:/usr/local/cuda
ENV CUDA_HOME /usr/local/cuda

RUN ldconfig

# Install python stuff
ADD requirements.txt /tmp/
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
RUN /usr/bin/python3 -m pip install Cython
RUN /usr/bin/python3 -m pip install tslearn

RUN cd /tmp && wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.29.1-linux64.tar.gz && \
    mv geckodriver /usr/bin/
