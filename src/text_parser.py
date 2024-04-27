from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader
from chromadb.utils import embedding_functions
import chromadb

from typing import List

class DBer:
    def __init__(self) -> None:
        self.embeddings = None
        self.vectordb = None

        self.chunk_size = 512
        self.overlap_percentage = 50
        
        pass

    def load_embedding_model(self):
        """
        Default embedding is 
            `sentence-transformers/all-MiniLM-L6-v2`
        """
        self.embeddings = embedding_functions.DefaultEmbeddingFunction()

    def load_vectordb(self):
        client = chromadb.HttpClient(host='localhost', port=8000)

        try: 
            client.heartbeat()
            print("chroma server is online. Creating client...")
        except ValueError:
            print("chroma server is not online")
        
        self.vectordb = client

    def vectorify_text(self, text_path: str) -> List[str]:
        overlap = int((self.overlap_percentage / 100) * self.chunk_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

        # Read text file
        txt = ""
        with open(text_path, "r") as file:
            txt = file.read()

        if txt == "":
            print("no content inside the text file")
            return
        
        docs = text_splitter.split_text(txt)
        return docs

    def vectorify_pdf(self, pdf_path: str) -> List[str]:
        overlap = int((self.overlap_percentage / 100) * self.chunk_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

        documents = PyPDFLoader(pdf_path).load()
        docs = text_splitter.split_documents(documents)
        return docs
    
    def collection_insert_text(self, vectorized: List[str], text_group: str, collection: str = "test_collection"):
        """
        If one wants to change the distance function of each embeddings,
            fyi "https://docs.trychroma.com/usage-guide#creating-inspecting-and-deleting-collections"

        """
        if self.vectordb is None:
            print("vector db not initialed. Use `load_vectordb` method")
            return

        if self.embeddings is None:
            print("embedding not loaded. Use `load_embedding_model` method")
            return
        
        # Create meta data and id list
        meta = list()
        ids = list()

        for i in range(len(vectorized)):
            meta.append({"title": text_group})
            ids.append(str(i + 1))
        
        clct = self.vectordb.get_collection(collection, embedding_function=self.embeddings)

        # Insert document
        clct.add(
            documents=vectorized,
            metadatas=meta,
            ids=ids,
        )

        return clct
