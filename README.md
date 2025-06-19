# RAG-FastAPI-Server

## Overview

**RAG-FastAPI-Server** is an API server for managing a lightweight Retrieval-Augmented Generation (RAG) database using FastAPI and PostgreSQL (with `pgvector` for vector similarity search). It enables storing text data with vector embeddings, efficient similarity search, and management of this data through a simple REST API. All core database logic is encapsulated in the `RAGLocal` class.

**[Article on Medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)** 

## Demo Video



https://github.com/user-attachments/assets/9fe633a2-a88e-454e-b84b-99c6cfd87e32



---

## Features

- **Create Indexed Tables**: Create new tables with an indexed vector column for embedding-based search.
- **Store RAG Items**: Insert text and its embedding vector into a specified table.
- **Similarity Search**: Retrieve the most similar documents to a given embedding (`cosine` or `euclidean` similarity).
- **RESTful API**: Interface with your database using HTTP POST requests, powered by FastAPI.
- **Database Handling**: Clean management of database connections and transactions.

---

## File Structure

```
.
├── main.py
└── RAGLocal/
    ├── __init__.py
    └── rag.py
```

### Main Components

#### `main.py`

- FastAPI app exposing the REST API.
- Loads environment variables for database credentials.
- Defines Pydantic models for payload structure.
- Endpoints:
  - `POST /rag/{dbname}/create/index`: Create a table + vector index.
  - `POST /rag/{dbname}/add/rag`: Insert a new document and its embedding.
  - `POST /rag/{dbname}/query`: Retrieve most similar documents to a query embedding.
- Instantiates a `RAGLocal` object per request to perform operations.

#### `RAGLocal/rag.py`

- Implements `RAGLocal`, the core database management class.
- Responsible for:
  - Connecting to PostgreSQL using `psycopg2` and handling credentials.
  - Creating tables and ivfflat vector indices for similarity search.
  - Inserting new records with a text and embedding.
  - Querying for top-K most similar embeddings, supporting "cosine" and "euclidean" similarity.
- Handles connections safely (using context manager and explicit close).
- Handles transaction commits and rollbacks.

#### `RAGLocal/__init__.py`

- Imports and exposes the `RAGLocal` class for package usage.

---

## API Usage

### 1. Create Index

Create a table for storing items with embeddings and a vector similarity index.

- **Endpoint**: `POST /rag/{dbname}/create/index`
- **Payload**:
  ```json
  {
    "name_index": "table_name",
    "content_name": "column_name",
    "embedding_dim": 1536,
    "type_index": "cos" // or "euclidean"
  }
  ```
- **Returns**: `{ "status": "index_created", "table": "..." }`

### 2. Add an Item

Add a new document and its embedding to a table.

- **Endpoint**: `POST /rag/{dbname}/add/rag`
- **Payload**:
  ```json
  {
    "table_name": "table_name",
    "content_column": "column_name",
    "content": "This is the text.",
    "embedding": [0.12, 0.85, ...] // Same length as embedding_dim
  }
  ```
- **Returns**: `{ "status": "item_added", "id": ... }`

### 3. Query for Similar Documents

Find the top K most similar items for the given embedding.

- **Endpoint**: `POST /rag/{dbname}/query`
- **Payload**:
  ```json
  {
    "table_name": "table_name",
    "content_column": "column_name",
    "query_embedding": [0.11, 0.87, ...],
    "top_k": 5,
    "type_index": "cos" // or "euclidean"
  }
  ```
- **Returns**:
  ```json
  {
    "status": "success",
    "results": [
      { "id": 1, "content": "...", "score": 0.9987 },
      ...
    ]
  }
  ```

---

## How It Works

- The server loads database credentials from a `.env` file.
- For each API call, a `RAGLocal` instance is created to perform the necessary DB operations.
- All vector operations rely on PostgreSQL with the `pgvector` extension for efficient vector storage and search using ivfflat indices.
- Similarity can be switched between cosine (`<#>`) and euclidean (`<->`) at index creation/query time.

---

## Key Python Functions

### In `main.py`:
- **get_rag**: Helper to instantiate `RAGLocal` with DB credentials.
- **rag_index**: Calls `create_index` to set up a new table and similarity index.
- **add_rag**: Calls `add_rag` to insert a text+embedding pair.
- **query_rag**: Calls `query` to retrieve the most similar items.

### In `RAGLocal/rag.py`:
- **RAGLocal.create_index**: Creates a table for your content and builds a vector similarity index.
- **RAGLocal.add_rag**: Inserts a new row with text and embedding; returns its DB id.
- **RAGLocal.query**: Queries the table for top-K matches, computing similarity scores.
- **Database Utilities** (connect, commit, close): Handle DB connections/transactions robustly.

---

## Requirements

- Python 3.8+
- PostgreSQL with `pgvector` extension enabled
- `psycopg2`, `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`

---

## Example: Running the API

1. Set up a PostgreSQL DB and install the `pgvector` extension.
2. Put your credentials in a `.env` file:
   ```
   DB_USER=your_user
   DB_PASSWORD=your_pass
   DB_HOST=localhost
   DB_PORT=5432
   ```
3. Start the server:
   ```
   uvicorn main:app --host 0.0.0.0 --port 5500
   ```
4. Use tools like `curl`, `httpie`, or Postman to interact with the API endpoints.

---

## Security Considerations

- Do **not** expose your database credentials publicly.
- Always use parameterized queries (as done in this code) to avoid SQL injection.
- Monitor who has access to your server and databases.

---
