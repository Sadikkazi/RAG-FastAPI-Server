import requests
from openai import OpenAI

client = OpenAI()

query_text = "Text string"

response = client.embeddings.create(
    input=query_text,
    model="text-embedding-3-small"
)

response_em = response.data[0].embedding

DB_NAME = "mi_rag"

payload = {
    "table_name": "textcontent",
    "content_column": "content",
    "query_embedding": response_em,
    "top_k": 5,
    "type_index": "cos"
}

url = f"http://127.0.0.1:5500/rag/{DB_NAME}/query"

resultados = requests.post(url=url, json=payload)

if resultados.status_code == 200:
    data = resultados.json()
    if data["status"] == "success":
        print(f"Query: {query_text}")
        for r in data["results"]:
            print(f"ID: {r['id']}, Content: {r['content']}, Score: {r['score']:.3f}")
    else:
        print("Error en la consulta:", data)
else:
    print(f"Error HTTP {resultados.status_code}: {resultados.text}")