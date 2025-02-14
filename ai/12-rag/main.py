from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
from database import Database
import os
import logging
import urllib.request

# Set logging level of transformers to ERROR to avoid unnecessary output
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "test")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "example")
RECREATE_INDEX = os.getenv("RECREATE_INDEX", "false").lower() == "true"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
QA_MODEL = "deepset/roberta-base-squad2"
TRANSLATE_ES_EN = "Helsinki-NLP/opus-mt-es-en"
TRANSLATE_EN_ES = "Helsinki-NLP/opus-mt-en-es"

def main(documents, queries):
    db = Database(
        api_key=PINECONE_API_KEY,
        region=PINECONE_ENV,
        index_name=INDEX_NAME,
        recreate_index=RECREATE_INDEX
    )

    translate_es_en = pipeline("translation_es_to_en", model=TRANSLATE_ES_EN)
    translate_en_es = pipeline("translation_en_to_es", model=TRANSLATE_EN_ES)
    qa_pipeline = pipeline("question-answering", model=QA_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    if RECREATE_INDEX:
        doc_embeddings = embed_model.encode(documents)

        progress_bar = tqdm(documents, desc=f"Inserting documents into Database")
        for i, doc in enumerate(progress_bar):
            db.insert([(str(i), doc_embeddings[i].tolist(), {"text": doc})])

    def translate_es_to_en(text):
        return translate_es_en(text)[0]["translation_text"]

    def translate_en_to_es(text):
        # Add prefix to text to ensure translation
        text = f"Translate: {text}"
        translation = translate_en_es(text)
        translation_text = translation[0]["translation_text"]
        translation_text = translation_text.split(": ")[1]
        return translation_text

    def retrieve_relevant_docs(query, top_k=2):
        query_embedding = embed_model.encode([query])[0].tolist()
        results = db.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        retrieved_docs = [match["metadata"]["text"] for match in results["matches"]]
        return [translate_es_to_en(doc) for doc in retrieved_docs]

    def generate_answer(query):
        retrieved_docs = retrieve_relevant_docs(query)
        context = " ".join(retrieved_docs)
        answer_en = qa_pipeline(question=translate_es_to_en(query), context=context)["answer"]
        return translate_en_to_es(answer_en)

    for query in queries:
        response = generate_answer(query)
        print(f"{query}\n - {response}\n")

def download_book():
    file_path = "./book.txt"
    # url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
    url = "https://raw.githubusercontent.com/busiris2014/7506Condor1C2014/refs/heads/master/datos2011/trunk/libros/J.K.%20Rowling%20-%20Harry%20Potter%203%20-%20El%20Prisionero%20de%20Azkaban.txt"

    if not os.path.exists(file_path):
        print("Downloading book...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    return text_data

def preprocess_text(text, chunk_size=512):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

if __name__ == "__main__":
    book = download_book()
    documents = preprocess_text(book)

    queries = [
        "Quien es Harry Potter?",
        "Quienes son sus mejores amigos?",
        "Que le pasa a Harry Potter?"
    ]

    main(documents, queries)
