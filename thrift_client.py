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

transport = TSocket.TSocket('localhost', 8080)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = KeyphraseModel.Client(protocol)
transport.open()

# x = {"id": "1", "title": "s3", "text": "txt"}

msg = client.predict([ttypes.Article("1", "s3", "txt")])
print(msg[0].keyphrase)

transport.close()