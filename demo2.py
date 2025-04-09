from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from backend.rag.embedder.embedder2 import Embedder2
from backend.rag.vectordb.vectordb2 import VectorDb2
from dotenv import load_dotenv
load_dotenv()
import os

# Ensure the persist_dir exists before using it.
STORAGE_PATH = "./chroma_db"
COLLECTION_NAME = "my_test_collection2"

embedding_model = OpenAIEmbedding(model="text-embedding-3-small",dimensions=256, batch_size=10)
# Set global LlamaIndex config
Settings.embed_model = embedding_model

# docs = [
#     Document(text="This Airbnb in New York has 3 bedrooms and a rooftop."),
#     Document(text="This listing is in fireplace and has white a pool."),
#     Document(text="This cozy cabin black in Denver includes a fireplace and mountain view."),
# ]
docs = [
    Document(text="you know what, prajwal rak is a hard working software engineer."),
    Document(text="my legs are hurting green."),
    Document(text="tany who is a bhai RAK is a dedicated mountain diver."),
]
def embed_and_store():
    embedder = Embedder2(STORAGE_PATH,COLLECTION_NAME,embedding_model)
    nodes = embedder.load_and_split_document(docs)
    embedder.embed_nodes_and_add_to_storage(nodes)

def query_from_vector_storage(query):
    vector_db = VectorDb2(STORAGE_PATH,COLLECTION_NAME)
    vector_db.query(query)

def retrive_from_vector_storage(query):
    vector_db = VectorDb2(STORAGE_PATH,COLLECTION_NAME)
    vector_db.retrieve(query)

query = "who is prajwal RaK"

#embed_and_store()
#query_from_vector_storage(query)

retrive_from_vector_storage(query)

##############################################################################################

# def embed_and_store():
#     """Embeds raw documents and stores them in ChromaDB."""
#     docs = [
#         Document(text="This Airbnb in New York has 3 bedrooms and a rooftop."),
#         Document(text="This listing is in fireplace and has white a pool."),
#         Document(text="This cozy cabin black in Denver includes a fireplace and mountain view."),
#     ]


#     vectordb = VectorDb(db_path="./chroma_test_db", collection_name="test_airbnb", persist_dir=PERSIST_DIR, load_index = False)
#     print(" -----------vectordb 1st ran!!")

#     # Embed and add to vector DB
#     embedded_nodes = embedder.process(docs)
#     print(f"📦 Number of nodes to index: {len(embedded_nodes)}")
#     vectordb.add_documents(embedded_nodes)

#     print("✅ Embedding complete and stored in ChromaDB.")


# def run_inference(query, top_k=2,load_index =True):
#     """Runs a similarity search on existing embedded documents."""
#     vectordb = VectorDb(db_path="./chroma_test_db", collection_name="test_airbnb", persist_dir=PERSIST_DIR,load_index= load_index)
#     results = vectordb.similarity_search_query(query, top_k=top_k)

#     print("🔍 Top results:")
#     for i, node in enumerate(results.source_nodes, 1):
#         print(f"{i}. {node.text}")


# query = "This Airbnb in New York?"

# embed_and_store()
# run_inference(query,True)
# #run_inference(query,True)