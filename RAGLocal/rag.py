import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from typing import List

class RAGLocal:
    def __init__(self, dbname: str, user: str, password: str, host: str, port: int = 5432, rag_multimodal=None):
        self.dsn = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
        }
        self.dbname = dbname
        self._conn = None

        self.rag_multimodal = rag_multimodal

    def connect(self):
        """Abre la conexión si no existe y la devuelve."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.dsn)
        return self._conn

    def cursor(self, dict_cursor: bool = False):
        """
        Devuelve un cursor.
        Si `dict_cursor=True`, el cursor devuelve filas como dicts.
        """
        conn = self.connect()
        if dict_cursor:
            return conn.cursor(cursor_factory=RealDictCursor)
        return conn.cursor()

    def commit(self):
        """Hace commit de la transacción actual."""
        if self._conn:
            self._conn.commit()

    def close(self):
        """Cierra cursor y conexión."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # Para usar con `with RAGLocal(...) as rag:`
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Si hubo excepción, la conexión hace rollback
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self.close()

    def create_index(
        self,
        table_name: str,
        content_column: str,
        embedding_dim: int = 1536,
        type_index: str = "cos"
    ):
        """
        Crea una tabla y un índice ivfflat sobre embedding,
        usando vector_cosine_ops o vector_l2_ops según type_index.
        """
        # Elegimos el operador según el tipo de índice
        op = "vector_cosine_ops" if type_index == "cos" else "vector_l2_ops"

        cur = self.cursor()
        # 1) Crear tabla si no existe
        cur.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    {content} TEXT NOT NULL,
                    embedding VECTOR({dim})
                );
            """).format(
                table=sql.Identifier(table_name),
                content=sql.Identifier(content_column),
                dim=sql.Literal(embedding_dim)
            )
        )
        # 2) Crear índice ivfflat
        cur.execute(
            sql.SQL("""
                CREATE INDEX IF NOT EXISTS {idx}
                ON {table}
                USING ivfflat (embedding {op})
                WITH (lists = 100);
            """).format(
                idx=sql.Identifier(f"{table_name}_emb_idx"),
                table=sql.Identifier(table_name),
                op=sql.SQL(op)
            )
        )
        self.commit()
        cur.close()

    def add_rag(
        self,
        table_name: str,
        content_column: str,
        content: str,
        embedding: List[float]
    ) -> int:
        """
        Inserta un nuevo registro en la tabla RAG y retorna su id.
        """
        cur = self.cursor()
        cur.execute(
            sql.SQL(
                "INSERT INTO {table} ({content}, embedding) VALUES (%s, %s) RETURNING id"
            ).format(
                table=sql.Identifier(table_name),
                content=sql.Identifier(content_column)
            ),
            (content, embedding)
        )
        new_id = cur.fetchone()[0]
        self.commit()
        cur.close()
        return new_id

    def query(
        self,
        table_name: str,
        content_column: str,
        query_embedding: List[float],
        top_k: int = 5,
        type_index: str = "cos"
    ) -> List[dict]:
        op = "<#>" if type_index == "cos" else "<->"
        cur = self.cursor(dict_cursor=True)

        sql_query = sql.SQL(
            "SELECT id, {content} AS content, embedding {op} %s::vector AS dist "
            "FROM {table} "
            "ORDER BY dist "
            "LIMIT %s;"
        ).format(
            content=sql.Identifier(content_column),
            op=sql.SQL(op),
            table=sql.Identifier(table_name)
        )

        cur.execute(sql_query, (query_embedding, top_k))
        rows = cur.fetchall()

        results = []
        for row in rows:
            dist = row["dist"]
            print("raw dist:", dist)
            if type_index == "cos":
                # Cosine: menor dist => más similar
                # Por los embeddings parece ser esta formula?
                score = max(0.0, ((1.0 - dist)/2))
            else:
                # Euclidean
                score = 1.0 / (1.0 + dist)
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": round(score, 3)
            })
        cur.close()
        return results

    def create_image_index(
        self,
        table_name: str,
        path_column: str = "path",
        embedding_dim: int = 512,
        type_index: str = "cos"
    ):
        """
        Crea tabla e índice para almacenar paths de imágenes y embeddings CLIP.
        """
        self.create_index(table_name, path_column, embedding_dim, type_index)

    def add_image(
        self,
        table_name: str,
        path_column: str,
        image_path: str
    ) -> int:
        """
        Usa la instancia rag_multimodal para obtener embedding de la imagen,
        y almacena junto al path.
        """
        if self.rag_multimodal is None:
            raise ValueError("Se requiere instancia rag_multimodal para agregar imágenes.")
        emb = self.rag_multimodal.get_embeddings(image_path).tolist()
        return self.add_rag(table_name, path_column, image_path, emb)

    def query_image(
        self,
        table_name: str,
        path_column: str,
        image_path: str,
        top_k: int = 5,
        type_index: str = "cos"
    ) -> List[dict]:
        """
        Recupera los paths de las imágenes más similares a la imagen de consulta.
        """
        if self.rag_multimodal is None:
            raise ValueError("Se requiere instancia rag_multimodal para consulta de imágenes.")
        query_emb = self.rag_multimodal.get_embeddings(image_path).tolist()
        results = self.query(table_name, path_column, query_emb, top_k, type_index)
        return results


