# 🔌 Circuit Agent Backend

This project powers the backend for a circuit-aware LLM agent that supports hardware component lookup, pin mapping, and semantic querying using vector embeddings and LLMs.

---

## 🚀 Getting Started

### 📦 Required Dependencies

To run this backend, make sure you have Python and install the following dependencies:

```bash
pip install llama-index-core
pip install llama-index-vector-stores-chroma
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-openai  # Optional: Only needed if you use OpenAI for embeddings

pip install chromadb
pip install python-dotenv

**✅ Tip:** It's recommended to use a virtual environment (`venv`) to manage dependencies and isolate your project from global packages.