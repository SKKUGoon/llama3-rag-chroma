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
    ra.load_llama()

    # Question 1
    question = "How much cash did Neutron rocket development burned during 2023?"
    ans, source = ra.augmented_generate(question=question)

    print("\n-----\n")
    print("Q1", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)

    # Question 2
    question = "Who are the main participants of this meeting?"
    ans, source = ra.augmented_generate(question=question)

    print("\n-----\n")
    print("Q2", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)

    # Question 3
    question = "What's the company's guidance for future quarter?"
    ans, source = ra.augmented_generate(question=question)

    print("\n-----\n")
    print("Q3", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)
