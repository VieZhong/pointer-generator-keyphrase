FROM tensorflow/tensorflow:1.4.0-gpu-py3

MAINTAINER zhangdunfeng<halo@hust.edu.cn>

COPY . /app

USER root

WORKDIR /app

RUN export LC_ALL=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    export LANGUAGE=en_US.UTF-8

RUN python -m pip  install --upgrade pip 
RUN pip install jieba==0.39 thrift==0.11.0 numpy==1.13.3 nltk==3.4

ENTRYPOINT ["python thrift_server.py 0.0.0.0 8084"]
