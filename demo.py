from llama_index.core import Document
from backend.rag.embedder.embedder import Embedder
from backend.rag.vectordb.vectordb import VectorDb
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(text="This Airbnb in New York has 3 bedrooms and a rooftop."),
    Document(text="This listing is in Miami and has a pool."),
    Document(text="This cozy cabin in Denver includes a fireplace and mountain view."),
]

# Initialize embedder and vectordb
embedder = Embedder()
vectordb = VectorDb(db_path="./chroma_test_db", collection_name="test_airbnb")

# Embed and add to vector DB
embedded_nodes = embedder.process(docs)
vectordb.add_documents(embedded_nodes)


# Custom query
query = "Which listing has a rooftop?"

# Run similarity search
results = vectordb.similarity_search_query(query, top_k=2)

# Print result text
print("Top results:")
for i, result in enumerate(results.response, 1):
    print(f"{i}. {result.text}")