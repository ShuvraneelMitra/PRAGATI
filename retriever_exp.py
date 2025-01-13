import os
import time
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.splitters import TokenCountSplitter
import pathway as pw
from dotenv import load_dotenv
import pprint
from DataIndex import DataIndex
import threading

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