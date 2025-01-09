import logging 
import sys
import os
import time
import string
from dotenv import load_dotenv

import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    filename='kdsh-2025.log')
logging.captureWarnings(True)
load_dotenv()

class DataIndex:
    def __init__(self, data_sources, embedder, splitter, **credentials):
        self.data_sources = data_sources
        self.credentials = credentials
        self.embedder = embedder
        self.splitter = splitter

        self.server = VectorStoreServer(*data_sources,
                                          embedder=embedder,
                                          splitter=self.splitter
                                        )
        self.client = VectorStoreClient(host="127.0.0.1",
                                        port=int(os.environ.get("PATHWAY_PORT"))
                                        )
        
    def run(self):
            self.server.run_server(host="127.0.0.1", port=int(os.environ.get("PATHWAY_PORT")), threaded=True, with_cache=False)

    def query(self, query):
        return self.client(query)


from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm import embedders
embedder = embedders.SentenceTransformerEmbedder(model="intfloat/e5-large-v2")
text_splitter = TokenCountSplitter()

data_sources = [pw.io.gdrive.read(object_id=os.environ.get("TMLR_OBJECT_ID"), 
                                  mode="static",
                                  service_user_credentials_file="credentials.json",
                                  with_metadata=True)]

@pw.udf
def binary_to_text(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass

res = data_sources[0].select(text=(data_sources[0].data))
with open("output.log", "w") as f:
    sys.stdout = f
    pw.debug.compute_and_print(res)

# pw.debug.compute_and_print(data_sources[0])
# d = DataIndex(data_sources, embedder, text_splitter)
# d.run()
# time.sleep(10)
# print(d.query("What is AI?"))
pw.run()


    

