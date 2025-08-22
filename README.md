# Smart-RAG: An Intelligent RAG System with a Dual-Retrieval Architecture

[![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-green)](https://shields.io/)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://shields.io/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE.md)

This project introduces **Smart-RAG**, an advanced Retrieval-Augmented Generation system designed to overcome the limitations of standard RAG frameworks. By leveraging a hybrid retrieval strategy and managing multiple, distinct knowledge sources, Smart-RAG aims to provide more accurate and context-aware answers to complex queries.

## 🎯 Problem Statement

Standard Retrieval-Augmented Generation (RAG) systems, while powerful, often face two key challenges in practical applications:

1.  **Single Retrieval Strategy:** Most RAG systems rely solely on vector similarity (semantic) search. This approach is effective for conceptual queries but often fails to retrieve documents containing specific keywords, function names, or technical terms where traditional lexical search would be superior.
2.  **Monolithic Knowledge Source:** When a knowledge base is built from diverse sources (e.g., a backend framework like FastAPI and an AI orchestration library like LangChain), a standard RAG system treats it as one large, undifferentiated pool of information. This can dilute retrieval relevance and makes it difficult to optimize search for domain-specific queries.

`Smart-RAG` is being developed to address these challenges by exploring a more intelligent and granular RAG architecture.

## ✨ Key Features

* **Dual-Retrieval Architecture:** Utilizes both **Elasticsearch** (for keyword search) and **Qdrant** (for semantic search) in parallel, providing two distinct and complementary retrieval pathways.
* **Multi-Source Knowledge Management:** Ingests and manages documents from different sources, tagging each chunk with metadata to enable source-specific filtering and routing.
* **Secure and Configurable:** All sensitive information (API keys) and environment-specific settings (database hosts, model names) are managed via a `.env` file, adhering to engineering best practices.
* **Decoupled API Service:** A robust backend built with **FastAPI** provides a clear, scalable interface for the core RAG logic.
* **Interactive Frontend:** A user-friendly web interface built with **Streamlit** allows for easy interaction and demonstration.

## 🏗️ System Architecture

The system follows a clean, layered architecture that decouples data processing, the core logic, and the user interface.

```
+----------------+      +-----------------+      +--------------------+
|                |      |                 |      |                    |
| Streamlit UI   +------>  FastAPI Backend  +------>  LLM Service      |
| (User Interface)|      | (Business Logic)|      | (e.g., OpenAI API) |
|                |      |                 |      |                    |
+----------------+      +-------+---------+      +--------------------+
                              |
                              |  (Parallel Retrieval)
            +-----------------+-----------------+
            |                                   |
+-----------v-----------+         +-----------v------------+
|                       |         |                        |
|   Qdrant              |         |   Elasticsearch        |
| (Semantic Search)     |         | (Keyword Search)       |
|                       |         |                        |
+-----------------------+         +------------------------+
```

## 🛠️ Tech Stack

* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Vector Database:** Qdrant
* **Full-Text Search Engine:** Elasticsearch
* **AI Orchestration:** LangChain (for document splitting)
* **Configuration:** python-dotenv
* **Environment & Deployment:** Docker, Docker Compose

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.9+
* Docker and Docker Compose

### 2. Installation & Setup

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)[Your-Username]/Smart-RAG.git
cd Smart-RAG

# 2. Create and configure the environment file
cp .env.example .env
# Now, edit the .env file with your actual API_KEY and other settings.

# 3. Start the database services
# This will launch Qdrant and Elasticsearch containers in the background.
docker-compose up -d

# 4. Install Python dependencies
pip install -r requirements.txt
```

### 3. Data Indexing

Before running the application, you need to process and index the source documents.

```bash
# This script will read, chunk, and index the documents into both Qdrant and Elasticsearch.
python indexing_pipeline.py
```

### 4. Running the Application

You need two separate terminals to run the backend and frontend services.

```bash
# In your first terminal, start the FastAPI backend:
uvicorn main:app --reload

# In your second terminal, start the Streamlit frontend:
streamlit run app.py
```
Now, open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 🗺️ Project Roadmap

This project is being developed in phases. Phase 1 is complete.

- [x] **Phase 1: Foundational RAG System & Dual Indexing**
    - [x] Collected and analyzed multi-source documents (FastAPI, LangChain).
    - [x] Implemented document chunking pipeline using LangChain.
    - [x] Set up Docker environment for Qdrant and Elasticsearch.
    - [x] Built a dual-indexing pipeline to populate both databases.
    - [x] Developed FastAPI backend and Streamlit frontend for a complete E2E workflow.
    - [x] Separated configuration from code using `.env`.

- [▶️] **Phase 2: Core Innovations - Smart Router & Adaptive Learning (In Progress)**
    - [ ] Develop a query analysis module for intelligent routing.
    - [ ] Design and implement an adaptive weighting mechanism based on user feedback.
    - [ ] Explore advanced chunking and fusion strategies.

- [ ] **Phase 3: Agentic System & Productionization (Planned)**
    - [ ] Introduce a lightweight agent for query decomposition and self-reflection.
    - [ ] Containerize the full application for deployment.
    - [ ] Establish basic monitoring and logging.


## 💡 Key Technical Decisions

A log of important technical decisions made during development.

* **Qdrant Point ID Management**
    * **Observation:** The Qdrant client library requires that each vector point's ID be either an integer or a UUID. Initial attempts to use custom string hashes failed due to this constraint.
    * **Solution:** An auto-incrementing integer ID is now used as the primary key for each point in Qdrant. A separate mapping layer is maintained in the application logic to link these integer IDs back to their original source document and chunk metadata.
    * **Trade-off:** This introduces a minor overhead of maintaining an ID map but ensures full compatibility and stability with the Qdrant database.

## 📄 License

Distributed under the MIT License. See `LICENSE.md` for more information.