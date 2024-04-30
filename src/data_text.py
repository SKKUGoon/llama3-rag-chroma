from parse import Vectorizer, RagTextSource

# How to insert raw text data into the vector database
# Define `RagTextSource` by giving file path and its title
# `_read()` method from `RagTextSource` will have automatically 
# procured the data from the source.
# 
# Load the `Vectorizer` class, load connection to vector database
# and embedding model. Insert `RagTextSource` and vectorify the text
# Insert the vectorified text into the collection.

if __name__ == "__main__":
    my_collection_nm = "test_collection"

    # Data
    data = RagTextSource(
        "./raw/test/rklb_q1_earnings.txt", 
        "Rocket lab 2024 Q1 Earnings"
    )
    data.read()

    # Vectorize
    vec = Vectorizer()
    vec.load_vectordb()
    vec.load_embedding_model()
    vec_result = vec.vectorify_text(data)

    vec.collection_insert_text(vec_result, data.title, my_collection_nm)
    

