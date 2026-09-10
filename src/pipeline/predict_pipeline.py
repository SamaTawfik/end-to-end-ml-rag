from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGPredictPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.chain = self._build_chain()

    def _build_chain(self):
        system_prompt = (
            "You are a helpful assistant. Answer the question strictly using the provided context below.\n"
            "If you do not know the answer, state that you do not know.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(self.retriever, question_answer_chain)

    def predict(self, query: str) -> str:
        response = self.chain.invoke({"input": query})
        return response.get("answer", "No answer generated.")