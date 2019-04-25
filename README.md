# 项目介绍

该项目是Co-occurrence Pointer Model(CoPM)的模型代码，可进行模型训练、验证、及测试，同时也是CoPM模型关键词提取Thrift Sever的启动工程。

# 数据集获取

原始数据集已上传至CSDC-FTP，目录为：ftp://zhongyw@192.168.88.210/engr/01.research/11.%E9%92%9F%E8%BF%9C%E7%BB%B4-%E4%B8%AD%E6%96%87%E8%AF%8D%E5%B5%8C%E5%85%A5%E4%B8%8E%E5%85%B3%E9%94%AE%E8%AF%8D%E6%8F%90%E5%8F%96/2.%E5%85%B3%E9%94%AE%E5%AD%97%E6%8F%90%E5%8F%96%EF%BC%88Keyword%20Extraction%EF%BC%89/4.%E5%AE%9E%E9%AA%8C%E6%95%B0%E6%8D%AE%E5%8F%8A%E4%BB%A3%E7%A0%81/%E6%95%B0%E6%8D%AE%E9%9B%86/

处理后的数据集在FTP目录：ftp://zhongyw@192.168.88.210/engr/01.research/11.%E9%92%9F%E8%BF%9C%E7%BB%B4-%E4%B8%AD%E6%96%87%E8%AF%8D%E5%B5%8C%E5%85%A5%E4%B8%8E%E5%85%B3%E9%94%AE%E8%AF%8D%E6%8F%90%E5%8F%96/2.%E5%85%B3%E9%94%AE%E5%AD%97%E6%8F%90%E5%8F%96%EF%BC%88Keyword%20Extraction%EF%BC%89/4.%E5%AE%9E%E9%AA%8C%E6%95%B0%E6%8D%AE%E5%8F%8A%E4%BB%A3%E7%A0%81/%E6%95%B0%E6%8D%AE%E9%9B%86/，运行工程前，需要将其解压至参数data_path与vocab_path指定的系统目录。

处理数据的代码，英文数据可参考data-preprocessing/make_datafiles_kp20k.py，中文数据可参考data-processing/make_datafiles_nssd.py。

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