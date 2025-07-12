import os
from pathlib import Path

try:
    import openai  # noqa: F401 – nur für API-Key-Check
except ImportError:
    openai = None  # type: ignore

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatLiteLLM  # Wrapper für LiteLLM
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

PDF_PATH = Path("BGB.pdf")
INDEX_DIR = Path("data/faiss_langchain")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

prompt= ChatPromptTemplate.from_messages([
    ("system", "Du bist ein juristischer Assistent, welcher die Fragen des Users anhand des gegebenen Kontextes ausgibt."
               "Beantworte die Fragen ausschließlich mit dem Wissen aus dem Kontext. Wenn du die Antwort daraus nicht "
               "extrahieren kannst antworte: Ich weiß es nicht."),
    ("human",  "Kontext:\n{context}\n\nFrage:{question}")
])


def build_or_load_vectorstore() -> FAISS:
    """Loads a vectorstore or creates a new one if not available"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # if faiss-index exists it gets loaded, if not it will be created
    if (INDEX_DIR / "index.faiss").exists():
        print("Lade bestehenden FAISS-Index …")
        vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {PDF_PATH}")

        # Faiss-index gets created: Pdf gets splitted in chunks and transformed into embeddings
        print("Erstelle neuen FAISS-Index …")
        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
        )
        docs = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def answer_question(question: str, vectorstore: FAISS) -> str:
    """answers a prompt question, when OpenAI-ApiKey is given, otherwise just gives top_k vectors"""
    if openai and openai_api_key:
        llm = ChatLiteLLM(
            model="gpt-4o-2024-11-20",
            temperature=0.0,
            api_key=openai_api_key
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs={"prompt": prompt},
            verbose=True,
        )
        return qa_chain.run({'query':str(question)})
    else:
        print(f"Kein OpenAI-API-Key vorhanden, gebe die top {TOP_K} Vektoren zurück. ")
        docs = vectorstore.similarity_search(question, k=TOP_K)
        return "\n\n".join(d.page_content for d in docs)


def main() -> None:
    vectorstore = build_or_load_vectorstore()

    print("RAG bereit! Stelle Deine Frage (oder 'exit' zum Beenden):")
    while True:
        try:
            question = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "bye"}:
            break
        if not question:
            continue
        answer = answer_question(question, vectorstore)
        print(f"\nAntwort:\n{answer}\n")


if __name__ == "__main__":
    main()
