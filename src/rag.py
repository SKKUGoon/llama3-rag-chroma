from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores.chroma import Chroma
import torch
from chromadb.utils import embedding_functions
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

class RetrieveAugment:
    sep = """\n\n--\n\n"""

    def __init__(self) -> None:
        self.max_chunk_in_context = 2
        
        self.llama = None
        self.vectordb = None
        self.embeddings = None
        pass

    def load_embedding_model(self):
        """
        Default embedding is
            `sentence-transformers/all-MiniLM-L6-v2`
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def load_vectordb(self):
        client = chromadb.HttpClient(host='localhost', port=8000)

        try:
            client.heartbeat()
            print("chroma server is online. Creating client...")
        except ValueError:
            print("chorma server is not online")

        self.vectordb = client
        print("vectordb ready")

    def load_tokenizer(self):
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_pipeline(self):
        self.llama = pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
        print("llama ready")

    def retrieve_context(self, query: str, collection_name: str = "test_collection"):
        """
        `R` of RAG.
        """
        if self.embeddings is None:
            return
        
        if self.vectordb is None:
            return

        client = Chroma(
            client=self.vectordb, 
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        sim_doc = client.similarity_search_with_relevance_scores(
            query=query,
            k=self.max_chunk_in_context
        )

        # client.
        return sim_doc, client
        # context = self.sep.join([doc.page_content for doc in sim_doc ])
        # ...