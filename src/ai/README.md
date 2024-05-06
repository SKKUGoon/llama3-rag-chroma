# RAG Example per class

## class `RetrieveAugment`

### Characteristics

### How to use

```python
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

```

...

## class `RetrieveAugmentChat`

### Characteristics

### How to use

```python
ra2 = RetrieveAugmentChat()
ra2.load_embedding_model()
ra2.load_vectordb()
ra2.load_llama()

# Question 0-1 - RAG
question = "How did the Rocket lab do financially during Q4 of the year 2023?"
ans = ra2.dynamic_generate(question)

print("Q0-1", question)
print("Answer\n")
print(ans)
print("---")

# Question 0-1 - RAG
question = "Does the rocket lab provide future guidance?"
ans = ra2.dynamic_generate(question)

print("Q0-1", question)
print("Answer\n")
print(ans)
print("---")

# Question 1 - Normal
question = "Hello. I'm Sangil. How are you?"
ans = ra2.dynamic_generate(question)

print("Q1", question)
print("Answer\n")
print(ans)
print("---")

# Question 2
question = "What was my name was again?"
ans = ra2.dynamic_generate(question)

print("Q1-1", question)
print("Answer\n")
print(ans)
print("---")
```
...