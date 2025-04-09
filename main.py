# === Import core LlamaIndex components ===
from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # free embedding model, comment if using openAI embeddings
# from llama_index.embeddings.openai import OpenAIEmbedding  # Optional: Uncomment if using OpenAI embeddings (cost credits)

# === Project-specific embedding and vector store classes ===
from backend.rag.embedder.embedder2 import Embedder2
from backend.rag.vectordb.vectordb2 import VectorDb2

# === Environment setup ===
from dotenv import load_dotenv
load_dotenv()
import os

# === Constants ===
STORAGE_PATH = "./chroma_db"                  # Path to persist vector storage (Chroma DB)
COLLECTION_NAME = "my_test_collection2"       # Name of the vector collection

# === Embedding Model Selection ===

# Option 1: HuggingFace Embedding (free, no cost, works offline)
# Note: This embedding model is suitable for storing and retrieving documents,
# but it does not support semantic "query()" functionality that depends on an LLM.
# Use only with "retrieve()" (similarity-based search without LLM).
# Embedding vector size = 378
embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Option 2: OpenAI Embedding (paid, requires API key)
# This model supports full semantic search including query() functionality using an LLM.
# Recommended if you plan to run cost-based, LLM-powered queries.
# Embedding vector size = 256
# NOTE: Uncomment both this block and the import statement  at the top to enable
#embedding_model = OpenAIEmbedding(model="text-embedding-3-small",dimensions=256, batch_size=10)


# === Apply selected embedding model to LlamaIndex global settings ===
Settings.embed_model = embedding_model

# === Sample documents for embedding and storage ===
# These documents simulate sensor/component pin mappings or generic test input.

# docs = [
#     Document(text="This Airbnb in New York has 3 bedrooms and a rooftop."),
#     Document(text="This listing is in fireplace and has white a pool."),
#     Document(text="This cozy cabin black in Denver includes a fireplace and mountain view."),
# ]
# docs = [
#     Document(text="you know what, prajwal rak is a hard working software engineer."),
#     Document(text="my legs are hurting green."),
#     Document(text="tany who is a bhai RAK is a dedicated mountain diver."),
#     Document(text="the pin mapping of A235 is pin1=vcc , pin2=gnd, pin3=terd, pin4=emp, pin5=sala, pin6=mur"),
#     Document(text="the pin mapping of A231 is pin1=mala , pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo")
# ]
# docs2 = [
#     Document(text="the pin mapping of B123 is pin1=mala , pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod"),
#     Document(text="the pin mapping of B919 is pin1=mala , pin2=GND4, pin3=vcc, pin4=BABA, pin5=bit, pin6=kolo,pin7=corod"),
#     Document(text="the pin mapping of B235 is pin1=vcc , pin2=gnd1, pin3=terd, pin4=emp, pin5=sala, pin6=mur"),
#     Document(text="the pin mapping of A333 is pin1=mala , pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod")
# ]
docs = [
    Document(text="the pin mapping of C11111 is pin1=mala , pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod, pin2=kala, pin3=vcc, pin4=arap, pin5=bit, pin6=kolo,pin7=corod"),
]


# === Pipeline: Embedding and Storing Documents ===
def embed_and_store():
    """
    Splits the documents into nodes and stores them in ChromaDB(our vector storage) using the selected embedding model.
    """
    embedder = Embedder2(STORAGE_PATH, COLLECTION_NAME, embedding_model)
    nodes = embedder.load_and_split_document(docs)
    embedder.embed_nodes_and_add_to_storage(nodes)

# === Pipeline: Query Vector Storage with LLM (expensive) ===
def query_from_vector_storage(query):
    """
    Executes a similarity query using the vector database.
    will use LLM internally to generate response, which could consume API credits.
    """
    vector_db = VectorDb2(STORAGE_PATH, COLLECTION_NAME)
    vector_db.query(query)

# === Pipeline: Retrieve Similar Documents (cheap, no LLM) ===
def retrive_from_vector_storage(query, show_similarity):
    """
    Retrieves similar documents from vector storage using pure embedding similarity.
    Does not invoke LLM, so it's free.
    """
    vector_db = VectorDb2(STORAGE_PATH, COLLECTION_NAME)
    vector_db.retrieve(query, show_similarity)


# === Sample query ===
query = "pin configuration of B919"


# === Usage: Uncomment only the step you need ===

# Step 1: Add documents to vector storage (run only once to upload data to vector storage)
# embed_and_store()

# Step 2: Query using LLM (incurs cost)
# query_from_vector_storage(query)

# Step 3: Retrieve similar documents from vector storage based on query without cost
retrive_from_vector_storage(query, show_similarity=True)

