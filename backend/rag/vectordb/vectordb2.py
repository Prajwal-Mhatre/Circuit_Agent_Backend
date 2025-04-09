import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

class VectorDb2:
    def __init__(self,storage_path, collection_name):
        # Initialize ChromaDB client and access the existing collection
        chroma_client = chromadb.PersistentClient(path=storage_path)
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        # Assign Chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load your index from the stored vectors
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


    def query(self,your_query):
        '''
        consumes credits: try to use it less frequently for debugging
        this function requires a llm and its api key
        it creates the query using rag knowledge and sends it to LLM using api key, and gets output
        '''
        # Create a query engine
        query_engine = self.index.as_query_engine()
        # Query the index
        response = query_engine.query(your_query)
        print(response)


    def retrieve(self,your_embedded_query,show_similarity):
        '''
        retrieves document with highest similarity score
        searches the vectorstore ( here chroma_db) and retrieves documents with highest similarity score
        '''
        retriever = self.index.as_retriever(similarity_top_k=4)
        retrieved_documents = retriever.retrieve(your_embedded_query)
        # Print the retrieved documents
        for doc in retrieved_documents:
            if show_similarity:
                print(doc)
            else:
                print(doc.text)

