from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from tqdm import tqdm

# another goal is to change embedding from openai to huggingfaceembedding if possible!)

class Embedder2:
    def __init__(self, storage_path, collection_name,embedding_model):

        # Initialize ChromaDB client and create a collection
        chroma_client = chromadb.PersistentClient(path=storage_path)
        self.chroma_collection = chroma_client.get_or_create_collection(collection_name)
        self.embedding_model = embedding_model


    def load_and_split_document(self,documents):
        # Load documents
        #documents = SimpleDirectoryReader("./data").load_data()

        # Split documents into nodes
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)   #node parcer, divides doc into nodes
        nodes = splitter.get_nodes_from_documents(documents)

        return nodes

    def embed_nodes_and_add_to_storage(self, nodes):

        # Embed nodes
        #embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        show_progress = True

        if show_progress:
            pbar = tqdm(total=len(nodes), desc="Embedding Progress", unit="node")

        embed_model = self.embedding_model
        for node in nodes:
            node.embedding = embed_model.get_text_embedding(node.text)

            assert node.embedding is not None, "❌ Embedding failed"
            if show_progress:
                pbar.update(1)

        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        vector_store.add(nodes)

        if show_progress:
            pbar.close()


########################
# you may require a index while embedding, below example creates the indes,
# note: it does not not require splitter because all the job is done by SimpledirectoryReader() , and they have set the global embedder in setting so they dont need to refrence it again and again!. 


# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core import Settings

# # global default
# Settings.embed_model = OpenAIEmbedding()

# documents = SimpleDirectoryReader("./data").load_data()

# index = VectorStoreIndex.from_documents(documents)

# # this above code chunk is done,

# now quering
# query_engine = index.as_query_engine()

# response = query_engine.query("query string")

########################


################################################################################################################

# # Initialize ChromaDB client and create a collection
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = chroma_client.get_or_create_collection("my_collection")

# # Load documents
# documents = SimpleDirectoryReader("./data").load_data()

# # Split documents into nodes
# splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
# nodes = splitter.get_nodes_from_documents(documents)

# # Embed nodes
# embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
# for node in nodes:
#     node.embedding = embed_model.embed(node.text)

# # Store nodes in ChromaDB
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# vector_store.add(nodes)

# # Create an index
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# # Query the index
# query_engine = index.as_query_engine()
# response = query_engine.query("Your query here")
# print(response)
