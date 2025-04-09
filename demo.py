from llama_index.core import Document
from backend.rag.embedder.embedder import Embedder
from backend.rag.vectordb.vectordb import VectorDb
from dotenv import load_dotenv
load_dotenv()
import os

# Ensure the persist_dir exists before using it.
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)
    print(f"Created persistence folder: {PERSIST_DIR}")

# Initialize embedder and vectordb
embedder = Embedder()
print(" -----------embedder ran!!")
def embed_and_store():
    """Embeds raw documents and stores them in ChromaDB."""
    docs = [
        Document(text="This Airbnb in New York has 3 bedrooms and a rooftop."),
        Document(text="This listing is in fireplace and has white a pool."),
        Document(text="This cozy cabin black in Denver includes a fireplace and mountain view."),
    ]


    vectordb = VectorDb(db_path="./chroma_test_db", collection_name="test_airbnb", persist_dir=PERSIST_DIR, load_index = False)
    print(" -----------vectordb 1st ran!!")

    # Embed and add to vector DB
    embedded_nodes = embedder.process(docs)
    print(f"📦 Number of nodes to index: {len(embedded_nodes)}")
    vectordb.add_documents(embedded_nodes)

    print("✅ Embedding complete and stored in ChromaDB.")


def run_inference(query, top_k=2,load_index =True):
    """Runs a similarity search on existing embedded documents."""
    vectordb = VectorDb(db_path="./chroma_test_db", collection_name="test_airbnb", persist_dir=PERSIST_DIR,load_index= load_index)
    results = vectordb.similarity_search_query(query, top_k=top_k)

    print("🔍 Top results:")
    for i, node in enumerate(results.source_nodes, 1):
        print(f"{i}. {node.text}")


query = "This Airbnb in New York?"

embed_and_store()
run_inference(query,True)
#run_inference(query,True)