# 🔌 Circuit Agent Backend

This project powers the backend for a circuit-aware LLM agent that supports hardware component lookup, pin mapping, and semantic querying using vector embeddings and LLMs.

---

## 🚀 Getting Started

### 📦 Required Dependencies

To run this RAG from backend, make sure you have Python and install the following dependencies:

```bash
pip install llama-index-core
pip install llama-index-vector-stores-chroma
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-openai  # Optional: Only needed if you use OpenAI for embeddings

pip install chromadb
pip install python-dotenv

```

**✅ Tip:** It's recommended to use a virtual environment (`venv`) to manage dependencies and isolate your project from global packages.

---

### 🔐 Environment Variables

> 📝 Make sure your `.env` file is configured with API keys if you're using OpenAI embeddings or LLM queries.

Create a `.env` file in your project root (same level as `main.py`) with the following content:

```env
OPENAI_API_KEY="your-openai-api-key"
```

🔒 **Important:** Ensure your `.env` file is listed in `.gitignore` to prevent accidental uploads.  
You can check this by opening `.gitignore` and verifying it includes:

```
.env
```

If it's not already there, add it manually.

---

### ▶️ Running the Project

Once dependencies are installed, run the `main.py` script.