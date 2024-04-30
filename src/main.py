from rag import RetrieveAugment


if __name__ == "__main__":
    ra = RetrieveAugment()
    ra.load_embedding_model()
    ra.load_vectordb()
    ra.load_llama()

    # Question 0
    question = "부동산 매입가 가격은?"
    ans, source = ra.augmented_generate(question=question, collection_name="test_real_estate")

    print("\n-----\n")
    print("Q0", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)

    # Question 0-1
    question = "부동산에 드는 총 비용은?"
    ans, source = ra.augmented_generate(question=question, collection_name="test_real_estate")

    print("\n-----\n")
    print("Q0-1", question)
    print("Answer")
    print(ans)
    print("\n-----\n")
    print("Source")
    print(source)

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
