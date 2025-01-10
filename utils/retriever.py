import logging 
import sys
import os
import time
import string
import threading
import pprint
from dotenv import load_dotenv

from custom_parser import CustomParse

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
    def __init__(self, data_sources, embeddeer_type, embedder_model="sentence-transformer", splitter_type="token-count", **credentials):
        if embeddeer_type == "sentence-transformer":
            self.embedder = embedders.SentenceTransformerEmbedder(model=embedder_model)
        if splitter_type == "token-count":
            self.splitter = TokenCountSplitter()
        self.data_sources = data_sources
        self.credentials = credentials
        self.parser = CustomParse()

        self.server = VectorStoreServer(*data_sources,
                                          embedder=self.embedder,
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

    def retrieve(self, query):
        """
        This is just a demo of how to use the DataIndex class, 
        all parameters will be changed in the eventual implementation
        """
        
        def get_data():
            response = d.query(self.query)
            with open('output.txt', 'w') as file:
                pprint.pprint(response, stream=file)
            
            for data in eval(response):
                yield(data)
    
        embedder = embedders.SentenceTransformerEmbedder(model="intfloat/e5-large-v2")
        text_splitter = TokenCountSplitter()

        data_sources = [pw.io.gdrive.read(object_id=os.getenv("NIPS_OBJECT_ID"), 
                                        service_user_credentials_file="credentials.json"
                                        )]

        d = DataIndex(data_sources, embedder, text_splitter)
        d.run()

        time.sleep(60)

        x = threading.Thread(target=get_data)
        x.start()

if __name__ == "__main__":
    d = DataIndex()
    d.retrieve("What is AI?")


    

