# Document Assistant with RAG

RAG-ассистент: отвечает на вопросы по вашим документам (PDF, DOCX, TXT), опираясь только на их содержимое. Индексирует файлы, ищет релевантные фрагменты и генерирует ответ через Ollama.

Настраиваемые параметры при инициализации: Размер чанка, overlap, число релевантных фрагментов (K)

## Технологии

- **Python 3.10+**, LangChain (core, community, text-splitters, ollama)
- **Эмбеддинги:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Векторный поиск:** FAISS
- **LLM:** Ollama (`mistral:7b`)
- **Документы:** PyPDF, docx2txt

## Запуск

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Нужен установленный и запущенный [Ollama](https://ollama.ai/) с моделью `mistral:7b` (`ollama pull mistral:7b`).

## Использование в коде

```python
from DocumentAssistant import DocumentAssistant

assistant = DocumentAssistant(chunk_size=900, overlap_size=200, top_k=4)
assistant.index_documents(["path/to/doc.pdf"])
answer = assistant.answer_query("Вопрос?")
assistant.save_results_to_json()
```
