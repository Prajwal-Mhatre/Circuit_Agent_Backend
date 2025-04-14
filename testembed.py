from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()

# Initialize the embedder (replace model and dimensions as you have them)
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256, embed_batch_size=1)

# Generate an embedding for a test string
embedding = embed_model.get_text_embedding("This is a test string.")

# Print the dimension of the returned vector
print("Embedding dimension:", len(embedding))