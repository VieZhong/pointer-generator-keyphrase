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

transport = TSocket.TSocket('192.168.101.4', 8085)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = KeyphraseModel.Client(protocol)
transport.open()

# x = {"id": "1", "title": "s3", "text": "txt"}
x = {
  "text": "数据新闻为传统的新闻报道注入了新的活力。如何运用大数据，做出社会大众喜闻乐见的新闻,是一个值得所有新闻从业人员和传媒人士深思熟虑的问题。本文在对数据新闻进行简要的阐述之余,还对数据新闻如何进行故事化处理提出了建议。",
  "id": "viezhong_for_test",
  "title": "浅谈数据新闻的数据化处理"
}
msg = client.predict([ttypes.Article(x["id"], "", x["text"])])
print(msg[0].keyphrases)

transport.close()
