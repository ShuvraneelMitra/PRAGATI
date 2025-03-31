import warnings
import logging
import uuid
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pdfparse.parse import ResearchPaperParser
from utils.chat import invoke_llm_langchain
import yaml
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
import chromadb

warnings.filterwarnings("ignore")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename="PRAGATI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()


class RAG:
    def __init__(self, pdf_path):
        logger.info(f"Initializing RAG with PDF: {pdf_path}")
        self.parser = ResearchPaperParser(pdf_path, output_dir="output", save=False)
        self.text = self.parser.process_document()["text"]

        logger.info(f"Successfully loaded document text")
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(
            "Initialized embedding model: sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1000
        Settings.chunk_overlap = 100
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["RAG_prompts"]
        logger.info(f"Loaded prompts from {prompts_path}")

    def prepare_documents_from_text(self, text):
        logger.info("Preparing documents from text")
        documents = []
        doc_id = str(uuid.uuid4())
        section_id = 1
        documents.append(
            {
                "content": text,
                "metadata": {
                    "source": "text",
                    "section_id": section_id,
                    "doc_id": doc_id,
                },
            }
        )
        logger.info(f"Created {len(documents)} documents from text")
        return documents

    def process_documents(self, docs):

        logger.info(f"Processing {len(docs)} documents into LlamaIndex format")

        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
        llama_nodes = []
        for doc in docs:
            for key, value in doc["content"].items():
                llama_doc = Document(text=value, metadata=doc["metadata"])
                nodes = splitter.get_nodes_from_documents([llama_doc])
                for i, node in enumerate(nodes):
                    node.metadata.update(
                        {
                            "chunk_id": f"{doc['metadata']['doc_id']}-chunk-{i}",
                            "chunk_index": i,
                            "total_chunks": len(nodes),
                        }
                    )

                llama_nodes.extend(nodes)

            logger.info(f"Created {len(llama_nodes)} total nodes from documents")
            return llama_nodes

    def create_db(self):
        logger.info("Creating vector database from documents")

        documents = self.prepare_documents_from_text(self.text)
        llama_nodes = self.process_documents(documents)
        collection_name = f"Randomness_{uuid.uuid4().hex[:8]}"
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        logger.info(f"Created Chroma collection: {collection_name}")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=llama_nodes, storage_context=storage_context)

        logger.info(f"Successfully created vector index with {len(llama_nodes)} nodes")
        return index

    def create_retriever(self, index, similarity_top_k=5):

        logger.info(f"Creating retriever with similarity_top_k={similarity_top_k}")

        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever

    def rag_query(self, query_text, retriever):
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query: {query_id} - '{query_text}'")

        retrieval_result = retriever.retrieve(query_text)
        logger.info(f"Retrieved {len(retrieval_result)} relevant nodes")

        context_parts = []
        source_documents = []

        for i, node in enumerate(retrieval_result):
            source_type = node.metadata.get("source", "unknown")
            doc_id = node.metadata.get("doc_id", "unknown")

            context_parts.append(
                f"[Document {i+1}] {source_type.capitalize()} {doc_id}: {node.text}"
            )

            source_documents.append(
                {"page_content": node.text, "metadata": node.metadata}
            )

        context = "\n\n".join(context_parts)
        prompt = self.prompts["human_message"].format(
            query_text=query_text, context=context
        )
        logger.debug(
            f"Generated prompt with context from {len(context_parts)} documents"
        )
        messages = [HumanMessage(content=prompt)]
        logger.info("Invoking LLM for response generation")
        updated_messages, input_tokens, output_tokens = invoke_llm_langchain(messages)
        logger.info(
            f"Generated response: {input_tokens} input tokens, {output_tokens} output tokens"
        )

        return {
            "query_id": query_id,
            "query": query_text,
            "result": updated_messages[-1].content,
            "source_documents": source_documents,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


if __name__ == "__main__":
    pdf_path = "C:/Users/MITRA/Desktop/Books/Tiny Machine Learning.pdf"
    logger.info(f"Starting RAG application with PDF: {pdf_path}")
    rag = RAG(pdf_path)
    index = rag.create_db()
    retriever = rag.create_retriever(index)
    while input("Enter 'q' to quit or anything else to ask your question: ") != "q":
        query_text = input("Enter your query: ")
        logger.info(f"User query: {query_text}")
        response = rag.rag_query(query_text, retriever)
        print(response)
