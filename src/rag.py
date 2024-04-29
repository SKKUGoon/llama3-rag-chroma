from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores.chroma import Chroma
import torch
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

from typing import List, Tuple


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

    def load_llama(self):
        self.llama = pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
        print("llama ready")

    def retrieve_context(self, query: str, collection_name: str = "test_collection") -> List[Tuple[Document, float]]:
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

        return sim_doc

    def process_retrieved(self, doc: List[Tuple[Document, float]]) -> str: 
        if len(doc) > 0:
            context = self.sep.join(
                [text.page_content for text, score in doc if score > 0.1]
            )  # _ is relevance score
            # context_less_relevant = self.sep.join(
            #     [text.page_content for text, score in doc if score <= 0.1]
            # )
            return context
        else:
            return "No relavent information were found for the user's question."
    
    def augmented_generate(self, question: str, collection_name: str = "test_collection"):
        if self.llama is None:
            print("LLM not loaded")
            return
        
        ctx = self.retrieve_context(question, collection_name)

        prefix_context = "From the context given below, answer the question of user \n"
        content_context = self.process_retrieved(ctx)

        message = [
            {
                "role": "system",
                "content": f"{prefix_context} {content_context}"
            },
            {
                "role": "user",
                "content": question,
            }
        ]

        prompt = self.llama.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.llama.tokenizer.eos_token_id,
            self.llama.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llama(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        return outputs[0]["generated_text"][len(prompt):], ctx

