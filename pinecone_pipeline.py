from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


@dag(
    dag_id="pinecone_pipeline",
    start_date=datetime(2026, 4, 25),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "Srija Taduri",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["pinecone", "embeddings", "vector-db"],
)
def pinecone_pipeline():
    @task()
    def download_and_prepare_data() -> str:
        records = [
            {
                "id": "doc_1",
                "text": "Pinecone is a vector database for similarity search.",
                "metadata": {"topic": "pinecone"},
            },
            {
                "id": "doc_2",
                "text": "Sentence transformers convert text into embeddings.",
                "metadata": {"topic": "embeddings"},
            },
            {
                "id": "doc_3",
                "text": "Airflow can orchestrate machine learning pipelines.",
                "metadata": {"topic": "airflow"},
            },
            {
                "id": "doc_4",
                "text": "Vector search helps find semantically similar content.",
                "metadata": {"topic": "search"},
            },
        ]

        output_file = "/tmp/pinecone_input.json"

        with open(output_file, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Created input file: {output_file}")
        print(f"Number of records: {len(records)}")
        return output_file

    @task()
    def create_pinecone_index() -> str:
        api_key = Variable.get("PINECONE-API-KEY")
        index_name = Variable.get("PINECONE_INDEX_NAME", default_var="pinecone-index-taduri")

        pc = Pinecone(api_key=api_key)

        existing = pc.list_indexes().names()
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Created index: {index_name}")
        else:
            print(f"Index already exists: {index_name}")

        return index_name

    @task()
    def embed_and_upsert(input_file: str, index_name: str) -> str:
        api_key = Variable.get("PINECONE-API-KEY")
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(input_file, "r") as f:
            records = json.load(f)

        texts = [r["text"] for r in records]
        embeddings = model.encode(texts).tolist()

        vectors = []
        for record, embedding in zip(records, embeddings):
            vectors.append(
                {
                    "id": record["id"],
                    "values": embedding,
                    "metadata": {
                        **record["metadata"],
                        "text": record["text"],
                    },
                }
            )

        index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors into {index_name}")
        return index_name

    @task()
    def search_pinecone(index_name: str) -> None:
        api_key = Variable.get("PINECONE-API-KEY")
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        model = SentenceTransformer("all-MiniLM-L6-v2")

        query_text = "What is Pinecone used for?"
        query_vector = model.encode([query_text])[0].tolist()

        result = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
        )

        print("Search results:")
        print(result)

    input_file = download_and_prepare_data()
    index_name = create_pinecone_index()
    upserted_index = embed_and_upsert(input_file, index_name)
    search_pinecone(upserted_index)


pinecone_pipeline()