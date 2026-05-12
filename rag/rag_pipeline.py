from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.custom_embedding import CustomEmbedding
import os  


class RAGPipeline:
    def __init__(self, api_key, base_url, data_path="data"):
        self.api_key = api_key
        self.base_url = base_url
        self.data_path = data_path

        self.vectorstore = self._build_vectorstore()

    def _load_documents(self):
        texts = []
        for file in os.listdir(self.data_path):
            if file.endswith(".txt"):
                with open(os.path.join(self.data_path, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
        return texts

    def _build_vectorstore(self):
        if os.path.exists("faiss_index"):
            return FAISS.load_local("faiss_index", CustomEmbedding(self.api_key, self.base_url),allow_dangerous_deserialization=True)
        docs = self._load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        split_docs = splitter.create_documents(docs)

        embeddings = CustomEmbedding(
            api_key=self.api_key,
            base_url=self.base_url
        )

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore

    def retrieve(self, query, k=3):
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    def build_context(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])