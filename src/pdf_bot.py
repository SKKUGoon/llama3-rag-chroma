from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class PDFParser:
    def __init__(self) -> None:
        self.embeddings = None
        pass

    def load_embedding_model(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        ...

    def load_vectordb(self):
        ...


