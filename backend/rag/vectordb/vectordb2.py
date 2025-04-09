import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

class VectorDb2:
    def __init__(self,storage_path, collection_name):
        """
        Initializes the vector database using ChromaDB as the backend.

        Args:
            storage_path (str): Directory where ChromaDB persistent data is stored.
            collection_name (str): Name of the vector collection (document group).

        Sets up:
            - A persistent ChromaDB client
            - A ChromaVectorStore for use with LlamaIndex
            - A StorageContext and VectorStoreIndex for querying or retrieving
        """

        # Initialize ChromaDB client and access the existing collection
        chroma_client = chromadb.PersistentClient(path=storage_path)
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        # Assign Chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

         # Create the storage context from the Chroma vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load your index from the stored vectors
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


    def query(self,your_query):
        """
        Comsumes API credits: Perform a semantic query using an LLM (costs API credits).

        Args:
            your_query (str): The user’s query string.

        This method uses LlamaIndex's Query Engine, which sends the query
        (enhanced via RAG) to an LLM. Ensure an embedding model + LLM is configured.
        Recommended only for high-level querying — not for frequent debugging.
        """
        # Create a query engine
        query_engine = self.index.as_query_engine()
        # Query the index
        response = query_engine.query(your_query)
        print(response)


    def retrieve(self,your_embedded_query,show_similarity):
        """
        Retrieve documents using similarity-based search (no LLM or API cost).

        Args:
            your_embedded_query (str): Query string to embed and compare against stored documents.
            show_similarity (bool): If True, show full document metadata including similarity scores.

        This method performs a fast nearest-neighbor search using only the vector embeddings.
        """
        retriever = self.index.as_retriever(similarity_top_k=4)
        retrieved_documents = retriever.retrieve(your_embedded_query)
        # Print the retrieved documents
        for doc in retrieved_documents:
            if show_similarity:
                print(doc)
            else:
                print(doc.text)

