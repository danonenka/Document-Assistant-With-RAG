import os
import urllib.request
import urllib.parse
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

    def __init__(self):
        self.top_k = 3
        self.llm = ChatOllama(model="mistral:7b", temperature=0.2)
        self.documents = []
        self.extracted_text_to_langchaindocs = []
        self.all_chunks = []
        self.download_dir = "downloaded_documents"
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.prompt_template = ChatPromptTemplate.from_template(
            "Используй только следующие фрагменты документов для ответа:\n"
            "Фрагменты: {retrieved_chunks}\n\n"
            "Вопрос: {query}\n"
            "Ответ:"
        )

    def index_documents(self, documents: list[str]):

        self.documents = documents
        for document in documents:
            self.extracted_text_to_langchaindocs.append(self.extract_text(document))

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  
        add_start_index=True 
        )
        for doc in self.extracted_text_to_langchaindocs:
            self.all_chunks.extend(text_splitter.split_documents(doc))

        self.vectorstore = FAISS.from_documents(self.all_chunks, self.embeddings)
        return f"Кол-во чанков{len(self.all_chunks)}"

    def answer_query(self, query: str) -> str:
        query_embedding = self.embeddings.embed_query(query)

        docs_with_scores = self.vectorstore.similarity_search_with_score_by_vector(
            query_embedding, 
            k=self.top_k
        )

        retrieved_docs = [doc for doc, _score in docs_with_scores]

        retrieved_chunks = "\n\n".join(f"[Фрагмент {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        )

        prompt = self.prompt_template.format_messages(
            retrieved_chunks=retrieved_chunks,
            query=query
        )
        response = self.llm.invoke(prompt)

        return response.content

    def _download_file(self, url: str, local_path: str) -> str:
        """Скачивает файл по URL и сохраняет локально"""
        os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
        return local_path

    def _get_file_path(self, document: str) -> str:
        """Получает локальный путь к файлу, скачивая его при необходимости"""

        if document.startswith(('http://', 'https://')):

            parsed_url = urllib.parse.urlparse(document)
            filename = os.path.basename(urllib.parse.unquote(parsed_url.path))
            
            local_path = os.path.join(self.download_dir, filename)
            if not os.path.exists(local_path):
                self._download_file(document, local_path)

            return local_path
        else:
            if not os.path.exists(document):
                raise FileNotFoundError(f"Файл не найден: {document}")
            return document

    def extract_text(self, document: str) -> list[Document]:
        """извлекает текст из файла (поддерживает URL и локальные пути)"""
        file_path = self._get_file_path(document)
        
        extension = Path(file_path).suffix.lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {extension}")
                
        documents = loader.load()
        
        return documents