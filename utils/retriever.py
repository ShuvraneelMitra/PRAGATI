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

    def query(self, query, k=3):
        return self.client.query(query, k=k)


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

    time.sleep(90)
    
    def f():
        with open("output.txt", 'w') as file:
            pprint(d.query("What is AI?", k=5), stream=file)

        

    x = threading.Thread(target=f)
    x.start()


class FileRetriever:
    def __init__(self, object_id, credentials_file="credentials.json", embedder_model="intfloat/e5-large-v2"):
        load_dotenv()
        self.embedder = embedders.SentenceTransformerEmbedder(model=embedder_model)
        self.splitter = TokenCountSplitter()
        self.data_sources = [
            pw.io.gdrive.read(object_id=object_id, service_user_credentials_file=credentials_file)
        ]
        self.index = DataIndex(
            data_sources=self.data_sources,
            embedder=self.embedder,
            splitter=self.splitter
        )

    def start_server(self):
        self.index.run()

    def retrieve_data(self, query, timeout=60):
        """
        Returns data using the same streaming principle from DataIndex.
        """
        # Start server for streaming
        self.start_server()
        # Wait to ensure server is ready
        time.sleep(timeout)
        # Query data
        return self.index.query(query)
    
# Usage
if __name__ == "__main__":
    object_id = os.getenv("NIPS_OBJECT_ID")

    def run_retriever():
        retriever = FileRetriever(object_id=object_id)
        query = "What is the best way to train a neural network?"
        response = retriever.retrieve_data(query)
        pprint.pprint(response)

    thread = threading.Thread(target=run_retriever)
    thread.start()
    print("Hello world")
    thread.join()


    

