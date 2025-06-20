from fastapi import FastAPI, HTTPException, UploadFile, Body, File
from pydantic import BaseModel
from typing import List
import os
from RAGLocal import RAGLocal
from dotenv import load_dotenv
from pathlib import Path
from RAGLocal import ImageRAG

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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

class CreateIndexImage(BaseModel):
    model_name: str = "openai/clip-vit-base-patch32"
    name_index: str
    content_name: str = "path"
    embedding_dim: int = 512 #Se deja porque CLIP tiene un embedding_dim de 512
    type_index: str = "cos"

class UploadImage(BaseModel):
    model_name: str = "openai/clip-vit-base-patch32"
    table_name: str
    content_name: str = "path"

class QueryRagImage(BaseModel):
    table_name: str
    content_column: str
    image_path: str
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

def get_rag_multimodal(dbname: str, model_name: str) -> RAGLocal:
    return RAGLocal(
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        rag_multimodal=ImageRAG(model_name=model_name)
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

@app.post("/rag/{dbname}/create/image/index")
async def create_multimodal_index(dbname: str, payload: CreateIndexImage):
    rag = get_rag_multimodal(dbname, payload.model_name)
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


@app.post("/rag/{dbname}/add/image/rag")
async def add_rag_multimodal(dbname: str, item: UploadImage = Body(...),
    file: UploadFile = File(...)):
    """Inserta un nuevo ítem en la tabla RAG y retorna el id generado."""
    rag = get_rag_multimodal(dbname, item.model_name)
    try:
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as out:
            content = await file.read()
            out.write(content)

        # 2) insertamos en la tabla RAG la ruta real
        new_id = rag.add_image(
            table_name=item.table_name,
            path_column=item.content_name,
            image_path=str(dest)  # Ruta completa al archivo
        )


        return {"status": "item_added", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        rag.close()

@app.post("/rag/{dbname}/image/query")
async def query_rag_image(dbname: str, q: QueryRagImage):
    pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=5500)