from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from RAGLocal import RAGLocal
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))

app = FastAPI(title="RAG Local API")

# Modelos Pydantic
class CreateIndex(BaseModel):
    name_index: str
    content_name: str
    embedding_dim: int = 1536
    type_index: str = "cos"

class AddRagItem(BaseModel):
    table_name: str
    content_column: str
    content: str
    embedding: List[float]

class QueryRag(BaseModel):
    table_name: str
    content_column: str
    query_embedding: List[float]
    top_k: int = 5
    type_index: str = "cos"

# Helper para obtener instancia RAGLocal
def get_rag(dbname: str) -> RAGLocal:
    return RAGLocal(
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

@app.post("/rag/{dbname}/create/index")
async def rag_index(dbname: str, payload: CreateIndex):
    """Crea tabla e índice en la base de datos especificada."""
    rag = get_rag(dbname)
    try:
        rag.create_index(
            table_name=payload.name_index,
            content_column=payload.content_name,
            embedding_dim=payload.embedding_dim,
            type_index=payload.type_index
        )
        return {"status": "index_created", "table": payload.name_index}
    finally:
        rag.close()

@app.post("/rag/{dbname}/add/rag")
async def add_rag(dbname: str, item: AddRagItem):
    """Inserta un nuevo ítem en la tabla RAG y retorna el id generado."""
    rag = get_rag(dbname)
    try:
        new_id = rag.add_rag(
            table_name=item.table_name,
            content_column=item.content_column,
            content=item.content,
            embedding=item.embedding
        )
        return {"status": "item_added", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        rag.close()

@app.post("/rag/{dbname}/query")
async def query_rag(dbname: str, q: QueryRag):
    """Recupera los documentos más similares y sus puntuaciones."""
    rag = get_rag(dbname)
    try:
        results = rag.query(
            table_name=q.table_name,
            content_column=q.content_column,
            query_embedding=q.query_embedding,
            top_k=q.top_k,
            type_index=q.type_index
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        rag.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=5500)