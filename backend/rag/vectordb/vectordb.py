import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

class VectorDb:
    def __init__(self, db_path=None, collection_name=None, persist_dir="./storage",load_index=False):
        try:
            print("db_path=",db_path)
            print("collection_name= ",collection_name)
            print("🔍 load_index =", load_index)
            print("🔍 persist_dir =", persist_dir)

            self.persist_dir = persist_dir

            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            # Create or get a collection
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            # Create ChromaVectorStore
            #self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection) #, persist_dir=self.persist_dir)
            # We will defer creating the storage_context until needed
            self.storage_context = None
            self.index = None
            # 3. Explicitly decide whether to load existing index
            if load_index:
                #try:
                print("🧠 Trying to load existing index...")              
                self._load_storage_context()
                self.index = load_index_from_storage(self.storage_context)
                print(" here is my self.storage_context i just got! in loading", self.storage_context)  

                #     # Optional sanity check
                #     if not self.index.docstore.docs:
                #         print("⚠️ Index loaded but is empty.")
                # except Exception:
                #     raise RuntimeError("❌ Failed to load index from storage. Make sure you embedded & persisted it.")
            else:
                print("📦 Skipping index load. Will create later.")
                #self.index = None  # Will be built later
            print("embedded vectordb running")    
        except Exception as e:
            print(f"❌ VectorDb init failed: {e}")
            raise

    def _load_storage_context(self):
        """Create storage context with persistence."""
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=self.persist_dir
        )

    def add_documents(self,nodes):

        #self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        #self.index.storage_context.persist()

        #did we add function for single node and multiple node insertion ?
        #below is code so nodes can be added multiple times!
        if self.index is None:
             # First time setup
             self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        else:
             # Append to existing index
             self.index.insert_nodes(nodes)
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(" here is my self.storage_context after adding nodes!!!!", self.storage_context)        

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

