from text_parser import DBer
from rag import RetrieveAugment


if __name__ == "__main__":
    # p = DBer()
    # p.load_vectordb()
    # p.load_embedding_model()
    # collection = p.vectordb.get_collection("test_collection")
    # print(collection.peek())

    ra = RetrieveAugment()
    ra.load_embedding_model()
    ra.load_vectordb()

    question = "How much cash did Neutron rocket development burned during 2023?"
    ctx, client = ra.retrieve_context(query=question)

    print(question)
    print("Answer")
    print(ctx)
