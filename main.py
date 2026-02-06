from dotenv import load_dotenv

load_dotenv()

from DocumentAssistant import DocumentAssistant


urls = ["https://storage.yandexcloud.net/sever-images/severstal/A9RD3D4.pdf", 
"https://storage.yandexcloud.net/sever-images/severstal/Polzovatelskoe_soglashenie.pdf",
"https://storage.yandexcloud.net/sever-images/severstal/University%20Success.docx"]

def main():
    document_assistant = DocumentAssistant()
    print(document_assistant.index_documents(urls))
    print(document_assistant.answer_query("Как зовут автора статьи So What Is a Business Model?"))
    print(document_assistant.answer_query("Кем была основана компания Dayton Dry Goods Company?"))


if __name__ == "__main__":
    main()