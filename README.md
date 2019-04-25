# 项目介绍

该项目是Co-occurrence Pointer Model(CoPM)的模型代码，可进行模型训练、验证、及测试，同时也是CoPM模型关键词提取Thrift Sever的启动工程。

# 数据集获取

原始数据集已上传至CSDC-FTP，目录为：

处理后的数据集在FTP目录：，运行工程前，需要将其解压至参数data_path与vocab_path指定的系统目录。

处理数据的代码，英文数据可参考data-processing/make_datafiles_kp20k.py，中文数据可参考data-processing/make_datafiles_nssd.py。

# 项目运行

## 模型训练

    python run_summarization.py --mode=train --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment --decode_only=False


## 模型验证

    python run_summarization.py --mode=eval --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment --decode_only=False

## 模型测试

    python run_summarization.py --mode=decode --data_path=/path/to/chunked/decode_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment --decode_only=False

## 启动 Thrift Server

在运行下面脚本时，请确保模型处于decode模式，并且decode_only与single_pass均为True。

    python thrift_server.py