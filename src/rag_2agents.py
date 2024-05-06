from transformers import AutoTokenizer, pipeline
from langchain.vectorstores.chroma import Chroma
import torch
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

from typing import List, Tuple, Final


class BColors:
    HEADER: Final = '\033[95m'
    OKBLUE: Final = '\033[94m'
    OKCYAN: Final = '\033[96m'
    OKGREEN: Final = '\033[92m'
    WARNING: Final = '\033[93m'
    FAIL: Final = '\033[91m'
    ENDC: Final = '\033[0m'
    BOLD: Final = '\033[1m'
    UNDERLINE: Final = '\033[4m'


class RetrieveAugmentChat:
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
        self.chat_history, self.chat_context_history = list(), list()
    
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
            print(BColors.OKGREEN, "[Status] healthy", BColors.ENDC)
            return True
        except ValueError:
            print(BColors.FAIL, "[Status] vector database down...", BColors.ENDC)
            return False

    def load_vectordb(self):
        client = chromadb.HttpClient(host='localhost', port=8000)

        try:
            client.heartbeat()
            print(BColors.OKGREEN, "[Status] chroma server is online. Creating client...", BColors.ENDC)
        except ValueError:
            print(BColors.FAIL, "[Status] chorma server is not online", BColors.ENDC)

        self.vectordb = client
        print(BColors.OKGREEN, "[Status] vectordb ready", BColors.ENDC)

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
        print(BColors.OKGREEN, "[Status] llama, terminator tokens, ready", BColors.ENDC)

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
        
    def process_chat_history(self) -> List[str]:
        hist_ls = list()
        for hist in self.chat_history:
            msg = f"{hist['role']}: {hist['content']}"
            hist_ls.append(msg)

        return hist_ls
    
    def process_chat_context_history(self) -> List[str]:
        hist_ls = list()
        for hist in self.chat_history:
            msg = hist['content']
            hist_ls.append(msg)
        
        return hist_ls
        
    def interprete(self, question: str):
        if self.llm is None or self.termination is None:
            print(BColors.FAIL, "[Status] LLM is not loaded properly", BColors.ENDC)
            return
        
        formatted_chat_history = self.sep.join(self.process_chat_history())
        formatted_context = self.sep.join(self.process_chat_context_history())
        
        interprete_system_prompt = f"""
You are analyzing the input question to determine its intent within the context of previous discussions.
Classify the question as either 'follow-up', 'new', or 'chat' based on the following criteria:

- 'follow-up': The question is directly connected to a topic previously discussed, seeking further details or clarification. It assumes prior knowledge from the chat history. If no such history exists, the question cannot be a 'follow-up'.

- 'new': The question introduces a completely new subject not previously mentioned in the chat history, prompting a fresh discussion.

- 'chat': The question is conversational, aiming to engage rather than to inform, often including greetings or casual remarks.

Assess the question's intent using the definitions above. If it's a 'follow-up', consider both the chat history and context to rephrase the question for deeper exploration.

Chat-history:\n{formatted_chat_history}

Context:\n{formatted_context}
"""
        
        message = [
            {
                "role": "system",
                "content": interprete_system_prompt,
            },
            {
                "role": "user",
                "content": f"Identify the user's intention when the user asked '{question}'.",
            }
        ]

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

        return outputs[0]["generated_text"][len(prompt):].strip().lower()

    def _normal_generate(self, question: str) -> str:
        message = [
            {
                "role": "user",
                "content": question,
            }
        ]
        self.chat_history.extend(message)

        prompt = self.llm.tokenizer.apply_chat_template(
            self.chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.llm(
            prompt,
            max_new_tokens=256,
            eos_token_id=self.termination,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        # Save the llm's answer to the chat history
        self.chat_history.extend([
            {
                "role": "assistant",
                "content": outputs[0]["generated_text"][len(prompt):],
            }
        ])

        return outputs[0]["generated_text"][len(prompt):]
    
    def _augmented_generate(self, question: str, collection_name: str, new: bool = True) -> str:
        if new or len(self.chat_context_history) <= 0:
            # Create new context related to the `question`.
            context = self.retrieve_context(question, collection_name)
            context_str = self.process_retrieved(context)

            self.chat_context_history = [{
                "role": "system",
                "content": f"{self.prompt_prefix}\nContext:\n{context_str}"
            }]

        user_message = [{
            "role": "user",
            "content": question
        }]
        self.chat_history.extend(user_message)
        
        prompt = self.llm.tokenizer.apply_chat_template(
            self.chat_context_history + self.chat_history,
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

        # Save the llm's answer to the chat history
        self.chat_history.extend([
            {
                "role": "assistant",
                "content": outputs[0]["generated_text"][len(prompt):],
            }
        ])
        print("Source", self.chat_context_history)
        return outputs[0]["generated_text"][len(prompt):]

    def dynamic_generate(self, question: str, collection_name: str = "test_collection") -> str:
        if self.llm is None or self.termination is None:
            print(BColors.FAIL, "[Status] LLM is not loaded properly", BColors.ENDC)
            return
        
        # Identify the user's intention
        intent = self.interprete(question)
        print(intent)

        return self._augmented_generate(question, collection_name, True)

        if "'new'" in intent:
            print(BColors.OKGREEN, f"[Status] Question's intention is 'new'", BColors.ENDC)
            return self._augmented_generate(question, collection_name, True)
        elif "'chat'" in intent:
            print(BColors.OKGREEN, f"[Status] Question's intention is 'chat'", BColors.ENDC)
            return self._normal_generate(question)
        elif "'follow-up'" in intent:
            print(BColors.OKGREEN, f"[Status] Question's intention is 'follow-up'", BColors.ENDC)
            return self._augmented_generate(question, collection_name, False)
        else:
            print(BColors.FAIL, f"[Status] unknown status {intent}", BColors.ENDC)
            return "(FAIL)"
