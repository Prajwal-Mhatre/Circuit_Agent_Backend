from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from tqdm import tqdm


class Embedder:
    def __init__(self, model_name="text-embedding-3-small", dimensions=256, batch_size=20, chunk_size=5000, chunk_overlap=200):
        self.embed_model = OpenAIEmbedding(
            model=model_name,
            dimensions=dimensions,
            embed_batch_size=batch_size,
        )
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Set global LlamaIndex config
        Settings.embed_model = self.embed_model

    def split_documents(self, llama_documents):
        """
        Takes a list of LlamaIndex Document objects and splits them into nodes.
        """
        return self.splitter.get_nodes_from_documents(llama_documents)

    def embed_nodes(self, nodes, show_progress=True):
        """
        Embeds each node in-place.
        """
        if show_progress:
            pbar = tqdm(total=len(nodes), desc="Embedding Progress", unit="node")

        for node in nodes:
            content = node.get_content(metadata_mode=MetadataMode.EMBED)
            node.embedding = self.embed_model.get_text_embedding(content)
            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        return nodes

    def process(self, llama_documents):
        """
        Convenience function to split + embed documents.
        Returns embedded nodes.
        """
        #split into smaller chunks
        nodes = self.split_documents(llama_documents)
        #embed each chunk
        return self.embed_nodes(nodes)
