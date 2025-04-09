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
        this function requires a llm and its api key
        this uses LLM api to generate output with rag knowledge
        '''
        # Create a query engine
        query_engine = self.index.as_query_engine()
        # Query the index
        response = query_engine.query(your_query)
        print(response)


    def retrieve(self,your_embedded_query,show_similarity):
        '''
        only retrieves documents based on similarity score
        '''
        # Create a query engine
        #query_engine = self.index.as_query_engine()
        retriever = self.index.as_retriever(similarity_top_k=4)
        #response = query_engine.query(your_query)
        retrieved_documents = retriever.retrieve(your_embedded_query)
        # Print the retrieved documents
        for doc in retrieved_documents:
            if show_similarity:
                print(doc)
            else:
                print(doc.text)


## what you may need while making a index
#     index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

# but the difference here is from_vector_store() and from_documents() , so i dont think we need to pass embedding model as parameter. 