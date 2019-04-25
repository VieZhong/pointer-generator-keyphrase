# coding: utf-8
"""
thrift_client.py
"""

import sys
from keyphrase import KeyphraseModel
from keyphrase import ttypes
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

transport = TSocket.TSocket('192.168.101.4', 8084)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = KeyphraseModel.Client(protocol)
transport.open()

# x = {"id": "1", "title": "s3", "text": "txt"}
x = {"text": "社会经济的发展,使基于计算机互联网的信息化技术普及程度越来越高,高校媒体肩负着新闻信息传播的重要使命,成为了当前引领高校思想的重要平台。为了促进教育事业的发展,推动和谐社会的建设,国家为宣传发展战略拟定出新的高度,旨在为高校宣传工作提供更好的发展空间。当前,大学生思想文化的多元化发展态势已成为新常态,传媒发展与时俱进也成为大势所趋。不过唯有如此,才能够有效发挥传媒的力量,守＂旧＂创＂新＂,切实履行好传媒的社会责任,致力于为当代社会思潮发展提供正确的导向。本文将以此出发,以视网融合为背景,浅谈纸媒新闻报道的新方向。", "id": "viezhong_for_test", "title": "探析视网融合下的高校纸媒新闻报道策略"}
msg = client.predict([ttypes.Article(x["id"], x["title"], x["text"])])
print(msg[0].keyphrases)

transport.close()
