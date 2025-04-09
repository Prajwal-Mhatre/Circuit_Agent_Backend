from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from backend.rag.embedder.embedder2 import Embedder2
from backend.rag.vectordb.vectordb2 import VectorDb2
from dotenv import load_dotenv
load_dotenv()
import os

# Ensure the persist_dir exists before using it.
STORAGE_PATH = "./chroma_db"
COLLECTION_NAME = "my_test_collection2"

embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
#uses embedding vector size = 378

# Set global LlamaIndex config
Settings.embed_model = embedding_model

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


def embed_and_store():
    embedder = Embedder2(STORAGE_PATH,COLLECTION_NAME,embedding_model)
    nodes = embedder.load_and_split_document(docs)
    embedder.embed_nodes_and_add_to_storage(nodes)

def query_from_vector_storage(query):
    vector_db = VectorDb2(STORAGE_PATH,COLLECTION_NAME)
    vector_db.query(query)

def retrive_from_vector_storage(query,show_similarity):
    vector_db = VectorDb2(STORAGE_PATH,COLLECTION_NAME)
    vector_db.retrieve(query,show_similarity)

query = "give me nothing "

#embed_and_store()
#query_from_vector_storage(query)

retrive_from_vector_storage(query,True)

##############################################################################################

# please confirm this
# it seems if 1 document is broken into 3 chunks, then while retriving , these 3 chunks are connected to each other and will be retrived all 3 at once for similarity found in any 3 of chunks, these 3 are treated as 1 document