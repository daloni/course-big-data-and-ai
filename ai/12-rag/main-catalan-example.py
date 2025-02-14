from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from database import Database
import os
import logging

# Set logging level of transformers to ERROR to avoid unnecessary output
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "test")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "example")
RECREATE_INDEX = os.getenv("RECREATE_INDEX", "false").lower() == "true"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
QA_MODEL = "deepset/roberta-base-squad2"
TRANSLATE_CA_EN = "Helsinki-NLP/opus-mt-ca-en"
TRANSLATE_EN_CA = "Helsinki-NLP/opus-mt-en-ca"

def main(documents, queries):
    db = Database(
        api_key=PINECONE_API_KEY,
        region=PINECONE_ENV,
        index_name=INDEX_NAME,
        recreate_index=RECREATE_INDEX
    )

    translate_ca_en = pipeline("translation_ca_to_en", model=TRANSLATE_CA_EN)
    translate_en_ca = pipeline("translation_en_to_ca", model=TRANSLATE_EN_CA)
    qa_pipeline = pipeline("question-answering", model=QA_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    if RECREATE_INDEX:
        doc_embeddings = embed_model.encode(documents)

        for i, doc in enumerate(documents):
            db.insert([(str(i), doc_embeddings[i].tolist(), {"text": doc})])

    def translate_ca_to_en(text):
        return translate_ca_en(text)[0]["translation_text"]

    def translate_en_to_ca(text):
        # Add prefix to text to ensure translation
        text = f"Translate: {text}"
        translation = translate_en_ca(text)
        translation_text = translation[0]["translation_text"]
        translation_text = translation_text.split(": ")[1]
        return translation_text

    def retrieve_relevant_docs(query, top_k=2):
        query_embedding = embed_model.encode([query])[0].tolist()
        results = db.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        retrieved_docs = [match["metadata"]["text"] for match in results["matches"]]
        return [translate_ca_to_en(doc) for doc in retrieved_docs]

    def generate_answer(query):
        retrieved_docs = retrieve_relevant_docs(query)
        context = " ".join(retrieved_docs)
        answer_en = qa_pipeline(question=translate_ca_to_en(query), context=context)["answer"]
        return translate_en_to_ca(answer_en)

    for query in queries:
        response = generate_answer(query)
        print(f"{query}\n - {response}\n")


if __name__ == "__main__":
    documents = [
        "Barcelona és una ciutat vibrant amb molta cultura.",
        "La Sagrada Família és una obra mestra arquitectònica.",
        "Gaudí va dissenyar molts edificis famosos a Catalunya."
    ]

    queries = [
        "Qui va dissenyar la Sagrada Família?",
        "Com es descriu Barcelona?",
        "Què va fer Gaudí a Catalunya?"
    ]

    main(documents, queries)
