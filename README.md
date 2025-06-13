# 📄 PDF Chatbot with LLMs

This project implements a **PDF Chatbot** that allows users to query PDF documents intelligently using large language models (LLMs) and vector search. The code explores different LLMs and techniques for document understanding and retrieval.

---
## ⚠️ Status
This project is experimental and under active development.
The code is intended for learning & experimentation. Features may change, and improvements are ongoing.
Feedback and contributions are welcome!

## 🚀 Features

* Upload and query PDF files.
* Leverages vector embeddings for document chunk search.
* Supports multiple LLM backends for answering questions:

  * **LLaMA 2**
  * **GPT-4**
  * **Google T5**
  * **OpenAI GPT**
* Embedding generation using:

  * **Sentence Transformers**
  * **Hugging Face Transformers**
* Vector database:

  * **Pinecone** for fast similarity search
* Framework:

  * **LangChain** for orchestration of LLMs, embeddings, and tools

---

## 🛠 Tech Stack

| Category            | Tools & Libraries                                         |
| ------------------- | --------------------------------------------------------- |
| **Language Models** | LLaMA 2, GPT-4, Google T5, OpenAI API                     |
| **Embeddings**      | Sentence Transformers, Hugging Face Transformers          |
| **Vector DB**       | Pinecone                                                  |
| **Framework**       | LangChain                                                 |
| **Other Libraries** | PyPDF2 / PyPDFLoader                                      |

---

## ⚙️ How It Works

1️⃣ PDF is uploaded and split into chunks.
2️⃣ Chunks are embedded into vector representations.
3️⃣ Vector store (Pinecone) indexes the embeddings.
4️⃣ On user query:

* Relevant chunks are retrieved by similarity search.
* Selected LLM generates an answer based on the retrieved content.

---

## 📌 Observations

* **Google T5** consistently provided the most accurate and context-aware responses.
* **LLaMA 2** worked well but showed variations depending on prompt and chunking.

---

## 📂 Files

* `pdf_chatbot.ipynb` – The main notebook with all code and experiments.

---

## ✅ Setup

```bash
pip install langchain sentence-transformers pinecone-client openai transformers
```

Set your API keys:

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
os.environ["PINECONE_API_KEY"] = "..."
```

---

## 💡 Future Work

* Add GUI / web interface for PDF upload & chat.
* Explore additional LLMs (e.g., Mistral, Claude).
* Optimize chunking and embedding strategies for longer documents.

---

## 🤖 Credits

Built with:

* [LangChain](https://github.com/langchain-ai/langchain)
* [Pinecone](https://www.pinecone.io/)
* [Hugging Face](https://huggingface.co/)
* [OpenAI](https://openai.com/)
* [Meta LLaMA](https://ai.meta.com/llama/)
