from rag_2agents import RetrieveAugmentChat


if __name__ == "__main__":
    ra2 = RetrieveAugmentChat()
    ra2.load_embedding_model()
    ra2.load_vectordb()
    ra2.load_llama()

    # Question 0-1 - RAG
    question = "What is Blackstone Real Estate Income Master Fund?"
    ans = ra2.dynamic_generate(question)
    
    print("Q0-1", question)
    print("Answer\n")
    print(ans)
    print("---")

    # Question 0-1 - RAG
    question = "When can Blackstone Real Estate Income Master Fund borrow money for re-financing purpose?"
    ans = ra2.dynamic_generate(question)
    
    print("Q0-1", question)
    print("Answer\n")
    print(ans)
    print("---")

    # # Question 0-1 - RAG
    # question = "How did the Rocket lab do financially during Q4 of the year 2023?"
    # ans = ra2.dynamic_generate(question)
    
    # print("Q0-1", question)
    # print("Answer\n")
    # print(ans)
    # print("---")

    # # Question 0-2 - RAG
    # question = "Does the rocket lab provide future guidance?"
    # ans = ra2.dynamic_generate(question)
    
    # print("Q0-2", question)
    # print("Answer\n")
    # print(ans)
    # print("---")

    # # Question 1 - Normal
    # question = "Hello. I'm Sangil. How are you?"
    # ans = ra2.dynamic_generate(question)
    
    # print("Q1", question)
    # print("Answer\n")
    # print(ans)
    # print("---")

    # # Question 2
    # question = "What was my name was again?"
    # ans = ra2.dynamic_generate(question)
    
    # print("Q1-1", question)
    # print("Answer\n")
    # print(ans)
    # print("---")
