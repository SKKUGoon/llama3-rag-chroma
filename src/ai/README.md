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
question = "Give me an overview on Apple's december quarter financial performance on 2024?"
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

## class `Vectorizer`

### Characteristic

How to insert raw text data into the vector database
Define `RagTextSource` by giving file path and its title
`_read()` method from `RagTextSource` will have automatically 
procured the data from the source.

Load the `Vectorizer` class, load connection to vector database
and embedding model. Insert `RagTextSource` and vectorify the text
Insert the vectorified text into the collection.

### How to use - Text file

```python

my_collection_nm = "test_collection"

# Data
# source_file = "./src/raw/test/rklb_q4_earnings.txt"
# source_file_title = "Rocket lab 2023 Q4 Earnings"

source_file = "./src/raw/test/googl_q1_earnings.txt"
source_file_title = "Google Inc 2024 Q1 Earnings"
data = RagTextSource(
    source_file, 
    source_file_title
)

# Vectorize
vec = Vectorizer()
vec.load_vectordb()
vec.load_embedding_model()
vec_result = vec.vectorify_text(data)

vec.collection_insert_text(vec_result, data.title, my_collection_nm)
```

### How to use - Table

```python
test_table = [
    ['', '원화 가치'],
    ['부동산 매입가', '212160'],
    ['취득세', '9945'],
    ['매입보수', '2122'],
    ['자문수수료', '480'],
    ['국민주택채권할인', '379'],
    ['중개보수', '1061'],
    ['담보 설정 비용', '589'],
    ['철거비용', '2500'],
    ['명도비용', '500'],
    ['기타비용', '1000'],
]


if __name__ == "__main__":
    my_collection_nm = "test_real_estate"

    # Data
    tb = RagStructTableSource(test_table)
    desc3d = tb.table_3d()
    desc3d = "\n\n--\n\n".join(desc3d)

    # Vectorize
    vec = Vectorizer()
    vec.load_vectordb()
    vec.load_embedding_model()
    vec_result = vec.vectorify_raw_text(desc3d)

    vec.collection_insert_text(vec_result, "부동산 취득 관련", my_collection_nm)
    print(vec.vectordb.get_collection(my_collection_nm).peek())
```