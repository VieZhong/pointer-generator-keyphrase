from keyphrase import KeyphraseModel
from keyphrase import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import json
import os
import subprocess

__HOST = '192.168.101.4'
__PORT = 8080

__DATA_PATH = '/tmp'
__INPUT_FILE = 'tmp_input.txt'
__OUTPUT_FILE = 'tmp_output.txt'


def write_to_input_file(article_list):
  with open(os.path.join(__DATA_PATH, __INPUT_FILE), "w", encoding='utf-8') as writer:
    for article in article_list:
      writer.write("%s\n" % json.dumps({"title": article["title"], "text": article["text"], "id": article["id"]}, ensure_ascii=False))


def read_from_output_file():
  results = []
  with open(os.path.join(__DATA_PATH, __OUTPUT_FILE), "r", encoding='utf-8') as lines:
    for line in lines:
      line = line.strip()
      if line:
        results.append(json.loads(line))
  return results
  


class KeyphrasesHandler(object):
  def predict(self, articles):
    article_list = [{"id": a.id, "title": a.title, "text": a.text} for a in articles]

    write_to_input_file(article_list)

    try:
      subprocess.check_call(["python", "run_summarization.py"])
    except subprocess.CalledProcessError:
      return []

    decode_results = read_from_output_file()

    return [ttypes.Keyphrase(r["id"], r["keyphrases"]) for r in decode_results]


if __name__ == '__main__':
  handler = KeyphrasesHandler()

  processor = KeyphraseModel.Processor(handler)
  transport = TSocket.TServerSocket(__HOST, __PORT)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()

  rpcServer = TServer.TSimpleServer(processor,transport, tfactory, pfactory)

  print('Starting the rpc server at', __HOST,':', __PORT)
  rpcServer.serve()
