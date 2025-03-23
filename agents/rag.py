import warnings
import logging
from pdfparse.parse import ResearchPaperParser
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid 
import os
from dotenv import load_dotenv
from utils.chat import invoke_llm_langchain
from langchain_core.messages import HumanMessage
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class RAG:
    def __init__(self, pdf_path):
        logger.info(f"Initializing RAG with PDF: {pdf_path}")
        # self.parser = ResearchPaperParser(pdf_path, output_dir="output", save=False) 

        text_file_path = "/home/naba/Desktop/PRAGATI/output/text_content.txt"
        try:
            with open(text_file_path, 'r') as file:
                text_content = file.read()
            self.text = text_content
        except FileNotFoundError:
            logger.error(f"Text file not found at {text_file_path}")
            self.text = None
        logger.info(f"Successfully parsed document")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #issue: cannot import name 'ImageNetInfo' from 'timm.data' 
        self.embeddings = embeddings
        logger.info("Initialized embedding model: text-embedding-004")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) 
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["RAG_prompts"]
        logger.info(f"Loaded prompts from {prompts_path}")
    
    def prepare_documents_from_text(self, text):
        logger.info(f"Preparing documents from texts")
        documents = []
        doc_id = str(uuid.uuid4())
        section_id = 1
        documents.append({
            "content": text,
            "metadata": {
                "source": "text", 
                "section_id": section_id,
                "doc_id": doc_id
            }
        })
        logger.info(f"Created {len(documents)} documents from text")
        return documents
    
    def prepare_documet(self, docs):
        logger.info(f"Splitting {len(docs)} documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = []
        for doc in docs:
            chunks = text_splitter.split_text(doc["content"])
            logger.debug(f"Split document {doc['metadata']['doc_id']} into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['metadata']['doc_id']}-chunk-{i}"
                split_docs.append({
                    "content": chunk, 
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
        documents = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in split_docs]
        
        logger.info(f"Created {len(documents)} total chunks")
        
        return documents
    
    def Create_db(self):
        logger.info("Creating vector database from documents")
        documents = self.prepare_documents_from_text(self.text)
        documents = self.prepare_documet(documents)
        collection_name = f"Randomness_{uuid.uuid4().hex[:8]}"
        logger.info(f"Creating Chroma vector database with collection name: {collection_name}")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            # persist_directory="./chroma_db"
        )
        logger.info(f"Successfully created vector database with {len(documents)} documents")
        return vectordb
    
    def Create_retriever(self, vectordb):
        logger.info("Creating retriever with k=5")
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        return retriever
    
    def rag_query(self, query_text, retriever):
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query: {query_id} - '{query_text}'")
        docs = retriever.get_relevant_documents(query_text)
        logger.info(f"Retrieved {len(docs)} relevant documents")
        context_parts = []
        for i, doc in enumerate(docs):
            source_type = doc.metadata['source']
            id_field = f"{source_type}_id"
            if id_field in doc.metadata:
                source_id = doc.metadata[id_field]
            else:
                source_id = "unknown"
            context_parts.append(
                f"[Document {i+1}] {source_type.capitalize()} {source_id}: {doc}"
            )
        context = "\n\n".join(context_parts)
        prompt = self.prompts['human_message'].format(query_text=query_text, context=context)
        logger.debug(f"Generated prompt with context from {len(context_parts)} documents")
        messages = [HumanMessage(content=prompt)]

        logger.info("Invoking LLM for response generation")
        updated_messages, input_tokens, output_tokens = invoke_llm_langchain(messages)
        logger.info(f"Generated response: {input_tokens} input tokens, {output_tokens} output tokens")
        return {
            "query_id": query_id,
            "query": query_text,
            "result": updated_messages[-1].content,
            "source_documents": docs,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

if __name__ == "__main__":
    pdf_path = "/home/naba/Desktop/PRAGATI/Tiny _ML_Things.pdf"
    logger.info(f"Starting RAG application with PDF: {pdf_path}")
    rag = RAG(pdf_path)
    vectordb = rag.Create_db()
    retriever = rag.Create_retriever(vectordb)
    
    while(input("Enter 'q' to quit: ") != 'q'):
        query_text = input("Enter your query: ")
        logger.info(f"User query: {query_text}")
        response = rag.rag_query(query_text, retriever)
        print(response["result"])
