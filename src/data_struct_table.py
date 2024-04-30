from parse import Vectorizer, RagStructTableSource

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