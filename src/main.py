from rag import RetrieveAugment


if __name__ == "__main__":
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
    question = "Who are the main participants of Apple's conference call?"
    ans, source = ra.augmented_generate(question=question)

    print("\n-----\n")
    print("Q2", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)

    # Question 3
    question = "Give me an overview on Apple's first quarter financial performance on 2024?"
    ans, source = ra.augmented_generate(question=question)

    print("\n-----\n")
    print("Q3", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)

    # Question 4
    question = "Give me an overview on Google's first quarter financial performance on 2024?"
    ans, source = ra.augmented_generate(question=question)

    print("\n-----\n")
    print("Q4", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)
