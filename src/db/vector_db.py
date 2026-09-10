import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorDBManager:
    def __init__(self, collection_name="rag_collection"):
        self.embeddings = OpenAIEmbeddings()
        self.collection_name = collection_name
        self.vector_store = None

    def create_vector_store(self, texts: list):
        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )
        return self.vector_store

    def get_retriever(self, k=3):
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})