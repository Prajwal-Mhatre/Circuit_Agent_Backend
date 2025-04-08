import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

class VectorDb:
    def __init__(self, db_path="./chroma_db", collection_name="default"):

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # Create or get a collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        # Create ChromaVectorStore
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        # Create a storage context
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create the index with the storage context
        self.index = None  # Will set later after adding nodes


    def add_documents(self,nodes):

        self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        self.index.storage_context.persist()

        #did we add function for single node and multiple node insertion ?
        #below is code so nodes can be added multiple times!
        # if self.index is None:
        #     # First time setup
        #     self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        # else:
        #     # Append to existing index
        #     self.index.insert_nodes(nodes)
        # self.index.storage_context.persist()        

    # retriver of our rag
    def similarity_search_query(self, query, top_k=3):
        """Perform similarity search using LlamaIndex's query engine."""
        if not self.index:
            raise RuntimeError("Index not initialized. Call add_nodes() first.")
        
        engine = self.index.as_query_engine(similarity_top_k=top_k)
        return engine.query(query)       


    def has_document(self,id):
        # decide if you want to implement it or not!
        pass

    def get_stats(self,):
        """Returns basic stats like number of vectors."""
        return {
            "collection_name": self.collection.name,
            "count": self.collection.count()
        }        

    def dev_clear(self):
        """Clear all documents in the collection (dev use)."""
        self.collection.delete(where={})        

