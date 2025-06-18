import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from typing import List

class RAGLocal:
    def __init__(self, dbname: str, user: str, password: str, host: str, port: int = 5432):
        self.dsn = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
        }
        self.dbname = dbname
        self._conn = None

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
        """
        Recupera los top_k registros más similares según el índice,
        y asigna un score entre 0 y 1.

        type_index: "cos" para cosine, "euclidean" para L2.
        """
        # Elegir operador de distancia PGVector
        op = "<#>" if type_index == "cos" else "<->"
        cur = self.cursor(dict_cursor=True)
        cur.execute(
            sql.SQL(
                f"SELECT id, {content_column} AS content, embedding {op} %s AS dist "
                f"FROM {table_name} ORDER BY dist LIMIT %s"
            ),
            (query_embedding, top_k)
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            dist = row["dist"]
            if type_index == "cos":
                score = max(0.0, 1.0 - dist)
            else:
                score = 1.0 / (1.0 + dist)
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": score
            })
        cur.close()
        return results

