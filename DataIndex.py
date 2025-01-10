import logging 
import sys
import os
import time
import string
import threading
from pprint import pprint
from dotenv import load_dotenv

from utils.custom_parser import CustomParse

import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm import embedders

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
        self.parser = CustomParse()

        self.server = VectorStoreServer(*data_sources,
                                          embedder=embedder,
                                          splitter=self.splitter,
                                          parser=self.parser
                                        )
        self.client = VectorStoreClient(host="127.0.0.1",
                                        port=int(os.getenv("PATHWAY_PORT"))
                                        )
        
    def run(self):
            self.server.run_server(host="127.0.0.1", 
                                   port=int(os.getenv("PATHWAY_PORT")), 
                                   threaded=True,
                                   with_cache=False)

    def query(self, query):
        return self.client.query(query)


def main():
    """
    This is just a demo of how to use the DataIndex class, 
    all parameters will be changed in the eventual implementation
    """

    embedder = embedders.SentenceTransformerEmbedder(model="intfloat/e5-large-v2")
    text_splitter = TokenCountSplitter()

    data_sources = [pw.io.gdrive.read(object_id=os.getenv("NIPS_OBJECT_ID"), 
                                    service_user_credentials_file="credentials.json"
                                    )]

    d = DataIndex(data_sources, embedder, text_splitter)
    d.run()

    time.sleep(60)
    
    def f():
        print("Waking up!")
        pprint((d.query("What is AI?")))

    x = threading.Thread(target=f)
    x.start()

if __name__ == "__main__":
    main()


    

