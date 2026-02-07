import json
import os
from pathlib import Path

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class DocumentAssistant:

    def __init__(self, chunk_size=1000, overlap_size=300, top_k=3):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.top_k = top_k
        self._results: list[dict] = []
        self._llm = ChatOllama(model="mistral:7b", temperature=0.1)
        self._vectorstore = None
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._prompt_template = ChatPromptTemplate.from_template(
            "Ты — помощник по работе с документами.\n\n"
            "Используй ТОЛЬКО информацию из приведённых ниже фрагментов.\n"
            "Если ответа в фрагментах нет — так и скажи.\n\n"
            "Фрагменты документов:\n"
            "{retrieved_chunks}\n\n"
            "Вопрос:\n"
            "{query}\n\n"
            "Правила:\n"
            "- Не используй внешние знания.\n"
            "- Не делай предположений.\n"
            "- Не дополняй ответ от себя.\n"
            "- Отвечай строго на русском языке.\n"
            "Ответ:"
        )

    def index_documents(self, documents: list[str]) -> None:
        docs = self._extract_text(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(docs)

        self._vectorstore = FAISS.from_documents(
            chunks,
            self._embeddings
        )

    def answer_query(self, query: str) -> str:
        if self._vectorstore is None:
            raise RuntimeError("Векторное хранилище не инициализировано. Вызовите index_documents сначала.")

        query_embedding = self._embeddings.embed_query(query)

        docs_with_scores = self._vectorstore.similarity_search_with_score_by_vector(
            query_embedding,
            k=self.top_k
        )

        retrieved_docs = [doc for doc, _ in docs_with_scores]

        retrieved_chunks = "\n\n".join(
            f"[Фрагмент {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        )

        prompt = self._prompt_template.format_messages(
            retrieved_chunks=retrieved_chunks,
            query=query
        )

        response = self._llm.invoke(prompt)
        answer = response.content
        self._results.append({"query": query, "answer": answer})
        return answer

    def _extract_text(self, documents: list[str]) -> list[Document]:
        all_docs = []
        for doc in documents:
            if not os.path.exists(doc):
                raise FileNotFoundError(f"Файл не найден: {doc}")

            extension = Path(doc).suffix.lower()

            if extension == ".pdf":
                loader = PyPDFLoader(doc)
            elif extension == ".docx":
                loader = Docx2txtLoader(doc)
            elif extension == ".txt":
                loader = TextLoader(doc, encoding="utf-8")
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {extension}")

            all_docs.extend(loader.load())

        return all_docs

    def save_results_to_json(self, path: str | Path = "results.json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, ensure_ascii=False, indent=2)