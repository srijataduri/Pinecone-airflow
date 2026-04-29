Pinecone Airflow Pipeline
Airflow pipeline integrating Pinecone vector database with sentence-transformers for embedding, ingestion, and search

## Overview
This project implements a Pinecone vector database pipeline using Apache Airflow.

## Steps Completed
1. Updated docker-compose.yaml to include:
   - sentence-transformers==3.1.1
   - pinecone==5.3.1
2. Restarted Docker containers
3. Configured Pinecone API key using Airflow Variables
4. Generated input JSON data for Pinecone
5. Created a Pinecone index
6. Generated embeddings using SentenceTransformers
7. Ingested vectors into Pinecone
8. Performed similarity search using Pinecone

## DAG Tasks
- download_and_prepare_data
- create_pinecone_index
- embed_and_upsert
- search_pinecone

## Technologies
- Apache Airflow
- Pinecone
- SentenceTransformers
- Python

## Files
- docker-compose.yaml
- dags/pinecone_pipeline.py
