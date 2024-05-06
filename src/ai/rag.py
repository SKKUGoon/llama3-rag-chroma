from transformers import AutoTokenizer, pipeline
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb
import torch

from typing import List, Tuple


class RetrieveAugment:
    sep = """\n\n--\n\n"""
    prompt_prefix = """
From the context given below, answer the question of user. 
If a question does not match the provided context, kindly advise the user to ask questions within the context of the document.
"""

    def __init__(self) -> None:
        self.max_chunk_in_context = 2
        
        self.llm = None
        self.vectordb = None
        self.embeddings = None
        self.termination = None

        # Stores chatting history
        self.user_name = None
        self.chat_history = list()

    def update_user_info(self, name: str) -> None:
        self.user_name = name

    def load_embedding_model(self):
        """
        Default embedding is
            `sentence-transformers/all-MiniLM-L6-v2`
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def health_check(self) -> bool:
        if self.vectordb is None:
            return False
        
        try:
            self.vectordb.heartbeat()
            print("healthy")
            return True
        except ValueError:
            print("vector database down...")
            return False

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
        self.llm = pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

        self.termination = [
            self.llm.tokenizer.eos_token_id,
            self.llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        print("llama, terminator tokens, ready")

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

    def process_retrieved(self, doc: List[Tuple[Document, float]], threshold: float = 0.05) -> str: 
        if len(doc) > 0:
            context = self.sep.join(
                [text.page_content for text, score in doc if score > threshold]
            )  # _ is relevance score
            # context_less_relevant = self.sep.join(
            #     [text.page_content for text, score in doc if score <= 0.1]
            # )
            return context
        else:
            return "No relavent information were found for the user's question."
    
    def build_sys_prompt(self, context: str):
        if self.user_name is not None:
            return f"The user's name is {self.user_name}. {self.prompt_prefix}\nContext:\n{context}"
        else:
            return f"{self.prompt_prefix}\nContext:\n{context}"
        ...

    def augmented_generate(self, question: str, collection_name: str = "test_collection"):
        if self.llm is None or self.termination is None:
            print("LLM is not loaded properly")
            return
        
        ctx = self.retrieve_context(question, collection_name)
        content_context = self.process_retrieved(ctx)

        # Because it's a RAG, Store the content history (system prompt) as well
        message = [
            {
                "role": "system",
                "content": f"{self.prompt_prefix}\nContext:\n{content_context}"
            },
            {
                "role": "user",
                "content": question,
            }
        ]
        self.chat_history.extend(message)

        prompt = self.llm.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.llm(
            prompt,
            max_new_tokens=256,
            eos_token_id=self.termination,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # Add the response from the assistant
        # Record the whole thing into `chat_history` attribute
        self.chat_history.extend([
            {
                "role": "assistant",
                "content": outputs[0]["generated_text"][len(prompt):],
            }
        ])

        return outputs[0]["generated_text"][len(prompt):], ctx

    def translated_response(self, language: str):
        ...

    def augmented_generate_multishot(self, shot: int = 3):
        ...
