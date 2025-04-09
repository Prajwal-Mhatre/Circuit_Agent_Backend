from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from tqdm import tqdm


class Embedder2:
    def __init__(self, storage_path, collection_name,embedding_model):

        # Initialize ChromaDB client and create a collection
        chroma_client = chromadb.PersistentClient(path=storage_path)
        self.chroma_collection = chroma_client.get_or_create_collection(collection_name)
        self.embedding_model = embedding_model


    def load_and_split_document(self,documents):


        # Split documents into nodes
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)   #node parcer, divides doc into nodes
        nodes = splitter.get_nodes_from_documents(documents)

        return nodes

    def embed_nodes_and_add_to_storage(self, nodes):


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


