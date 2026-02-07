from dotenv import load_dotenv

load_dotenv()

from DocumentAssistant import DocumentAssistant

files = [
    "downloaded_documents/A9RD3D4.pdf",
    "downloaded_documents/Polzovatelskoe_soglashenie.pdf",
    "downloaded_documents/University Success.docx",
]

queries = [
    "Кем была основана компания Dayton Dry Goods Company?",
    "Кем был Peter Drucker?",
    "What is bussines model?",
    "В чём заключается принцип законности политики АО «СЕВЕРСТАЛЬ МЕНЕДЖМЕНТ»?",
    "Опиши кратко два направления деятельности АО «СЕВЕРСТАЛЬ МЕНЕДЖМЕНТ»",
    "Кому может предоставляться доступ к услуге по технологии Ethernet?",
    "При доступе в интернет можно производить рассылку СПАМа, вредоносных программ?",
    "Сколько компонентов имеет Business Model Canvas?",
    "В чём заключается цель политики АО «СЕВЕРСТАЛЬ МЕНЕДЖМЕНТ»?",
]


def main():
    document_assistant = DocumentAssistant(900, 200, 4)
    document_assistant.index_documents(files)
    for query in queries:
        document_assistant.answer_query(query)
    document_assistant.save_results_to_json()


if __name__ == "__main__":
    main()