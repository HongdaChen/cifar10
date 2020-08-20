FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install torchvision  \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==2.0.0a0 \
    pip install tensorboardX \
    && rm -rf ~/.cache/pip 

ENV GLOO_SOCKET_IFNAME=eth0
ENV NCCL_SOCKET_IFNAME=eth0

WORKDIR /work
RUN python -c "from torchvision import datasets;datasets.CIFAR10('data', download=True)"
COPY utils ./utils
COPY models ./models
