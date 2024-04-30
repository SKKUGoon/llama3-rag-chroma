from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader
from chromadb.utils import embedding_functions
import chromadb
import fitz  # PyMuPDF

from typing import List, Tuple
import warnings


class RagTextSource:
    def __init__(self, path: str | None = None, title: str | None = None):
        self.path = path

        if self.title is None:
            self.title = "Untitled Text Source"
        else:
            self.title = title

        self.language_data = None
        self._read()
    
    def _read(self):
        if self.path is None: 
            return

        try:
            with open(self.path, "r") as file:
                txt = file.read()

                if txt == "":
                    print("no content inside the given text file")
                    return
                self.language_data = txt

        except FileNotFoundError:
            print(f"no file named {self.path}")
            return
        

class RagPDFTableSource:
    def __init__(self, path: str = None, title: str = None) -> None:
        self.path = path

        if self.title is None:
            self.title = "Untitled Text Source"
        else:
            self.title = title

    def pdf_table_raw(self):
        if self.path is None:
            return dict()

        # Open the provided PDF File
        doc = fitz.open(self.path)
        tables = dict()

        # Iterate through each page in the PDF
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            tables[page_num + 1] = list()
            
            # Search for tables and extract data
            for b in page.get_text("dict")["blocks"]:
                table = list()
                
                if b["type"] == 0:  # Text block
                    for l in b["lines"]:
                        row = list()
                        for s in l["spans"]:
                            if s["text"].strip():  # ensuring there is text
                                row.append(s["text"].strip())
                        if row:
                            table.append(row)

                not_table_condition1 = not (len(table) == 1 and len(table[0]) == 1)
                not_table_condition2 = len(table) > 0


                if not_table_condition1 and not_table_condition2:
                    # Not a table like structure
                    tables[page_num + 1].append(table)

        return tables

    def pdf_table_unstructured(self):
        ...


class RagStructTableSource:
    # Small table - Excel table
    def __init__(self, table_data: List[List]) -> None:
        if len(table_data) == 0:
            print("no data")
            return
        
        self.size = self._table_size(table_data)
        self.data = table_data

    @staticmethod
    def _table_size(table_data: List[List]) -> Tuple[int]:
        vert = len(table_data)
        hori = len(table_data[0])
        
        if vert > 50:
            warnings.warn(f"Table size is ({vert}, {hori}). Please consider using methods other than describing it in text")

        for rows in table_data:
            assert len(rows) == hori, "different column length"

        return (vert, hori)

    def table_2d(self, 
                 axis: int = 0, 
                 data_horizontal_offset: int = 1, 
                 data_vertical_offset: int = 1) -> List[str]:
        """
        :param axis: 0 if vertical, 1 if horizontal
        :param vertical_column_offset: Where does the data begin? when axis=0. Default value 1
        :param horizontal_column_offset: Where does the data begin? when axis=1. Default value 1


        axis = 0
        | column_name1 | data value |
        | column_name2 | data value |
        | column_name3 | data value |
                        ^^^^^^^^^^^^
                        data_horizontal_offset = 1. Data start from '1' column position
        
        axis = 1
        | column name1 | column name2 | column name3 |
        |  data value  |  data value  |  data value  | < data_vertical_offset = 1. Data start from '1' row position
        """
        desc: List[str] = list()
        assert axis == 0 or axis == 1

        if axis == 0:
            for row in self.data:
                row_desc = ", ".join([str(v) for v in row[data_horizontal_offset:]])
                row_desc_prefix = f"For '{row[0]}', the value is"
                desc.append(f"{row_desc_prefix} '{row_desc}'")

            return desc
        else:
            cols = self.data[0]
            desc = list()
            for row in self.data[data_vertical_offset:]:
                row_desc_elem: List[str] = list()
                for col_name, value in zip(cols, row):
                    row_desc_elem.append(f"The value of '{col_name}' is '{value}'")
                
                row_desc = " and ".join(row_desc_elem)
                desc.append(row_desc)

    def table_3d(self, 
                 x_axis_loc: int = 0, 
                 y_axis_loc: int = 0, 
                 data_horizontal_offset: int = 1, 
                 data_vertical_offset: int = 1) -> List[str]:
        """
        :param x_axis_offset: Where does the data on x_axis begin? Default value 1
        :param y_axis_offset: Where does the data on y_axis begin? Default value 1
        |          |  column x  |
        | column y | data value |
        """
        cols = self.data[x_axis_loc][data_horizontal_offset:]
        desc: List[str] = list()
        for row in self.data[data_vertical_offset:]:
            assert len(cols) == len(row[data_horizontal_offset:])
            
            row_desc_elem: List[str] = list()
            for col_name, value in zip(cols, row[data_horizontal_offset:]):
                row_desc_elem.append(f"the value of '{col_name}' is '{value}'")
            
            row_desc = " and ".join(row_desc_elem)
            row_prefix = f"For '{row[y_axis_loc]}'"

            desc.append(f"{row_prefix} {row_desc}")
        
        return desc


class RagSQLSource:
    def __init__(self) -> None:
        ...

class Vectorizer:
    def __init__(self) -> None:
        self.embeddings = None
        self.vectordb = None

        self.chunk_size = 512
        self.overlap_percentage = 50

    def load_embedding_model(self):
        """
        Default embedding is 
            `sentence-transformers/all-MiniLM-L6-v2`
        """
        self.embeddings = embedding_functions.DefaultEmbeddingFunction()

    def load_vectordb(self):
        client = chromadb.HttpClient(host='localhost', port=8000)

        try: 
            client.heartbeat()
            print("chroma server is online. Creating client...")
        except ValueError:
            print("chroma server is not online")
        
        self.vectordb = client

    def vectorify_text(self, source: RagTextSource) -> List[str]:
        overlap = int((self.overlap_percentage / 100) * self.chunk_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

        # Read text file
        if source.language_data is None:
            return []  # Empty list
        
        docs = text_splitter.split_text(source.language_data)
        return docs
    
    def vectorify_raw_text(self, source: str) -> List[str]:
        overlap = int((self.overlap_percentage / 100) * self.chunk_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

        docs = text_splitter.split_text(source)
        return docs

    def vectorify_pdf_text(self, pdf_path: str) -> List[str]:
        overlap = int((self.overlap_percentage / 100) * self.chunk_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

        documents = PyPDFLoader(pdf_path).load()
        docs = text_splitter.split_documents(documents)
        return docs

    def collection_insert_text(self, vectorized: List[str], text_title: str, collection: str = "test_collection"):
        """
        If one wants to change the distance function of each embeddings,
            fyi "https://docs.trychroma.com/usage-guide#creating-inspecting-and-deleting-collections"

        """
        if self.vectordb is None:
            print("vector db not initialed. Use `load_vectordb` method")
            return

        if self.embeddings is None:
            print("embedding not loaded. Use `load_embedding_model` method")
            return
        
        # Create meta data and id list
        meta = list()
        ids = list()

        for i in range(len(vectorized)):
            meta.append({"title": text_title})
            ids.append(str(i + 1))
        
        clct = self.vectordb.get_or_create_collection(collection, embedding_function=self.embeddings)

        # Insert document
        clct.add(
            documents=vectorized,
            metadatas=meta,
            ids=ids,
        )

        return clct
