# 🛡️ VISION: Violence, Health and Society – RAG QnA Platform

A Retrieval-Augmented Generation (RAG) platform for the VISION research consortium, funded by the UK Prevention Research Partnership (UKPRP). This app supports advanced QnA over sensitive violence and health data, with multi-tenant workspaces for central government, local government, and third sector partners.

VISION brings together public bodies, universities, and third sector organisations to reduce the harms to health caused by violence by improving the data that underpins theory, policy, and professional practice. The platform enables secure, isolated QnA over uploaded documents (PDF/CSV), leveraging semantic caching and cross-encoder reranking for high-quality answers.

Consortium partners include City, University of London, King’s College London, UCL, Lancaster, Bristol, Warwick, UCLan, and Public Health Wales. The platform integrates data from police, justice, health, third sector services, and national surveys, using advanced computing techniques for data coding and integration.

---

## 🚀 Features
- Upload and process PDF or CSV files for QnA
- Multi-tenant: three isolated workspaces (central government, local government, third sector)
- Semantic cache for fast repeated QnA
- Cross-encoder reranking for better answer relevance
- Secure, privacy-aware document handling
- Powered by Ollama, ChromaDB, Redis, and Streamlit

---

## 🤖 Prerequisites
- [Ollama](https://ollama.dev/download) (for local LLM inference)
- Python 3.11+
- Redis (for semantic cache) **MUST be redis==4.5.5** (do not use redis 5.x or 6.x)
- ChromaDB (vector store, runs locally)

---

## 🛠️ Setup

Create a virtual environment and install dependencies:

```sh
make setup
```

---

## ⚡️ Running the App

```sh
make run
```

- The app will be available at the URL shown in your terminal (e.g., http://localhost:8501)
- Share the "Network URL" for team testing on your LAN

---

## 🗂️ Configuring Workspaces

Workspaces are defined in `workspaces.yaml`:

```yaml
workspaces:
  - Central Government
  - Local Government
  - Third Sector
```

- Each workspace is isolated for its respective partner group.
- Edit this file to adjust workspace names as needed.
- No code changes required—just update the YAML and restart the app.

---

## 🧹 Linters and Formatters

Check for linting rule violations:

```sh
make check
```

Auto-fix linting violations:

```sh
make fix
```

---

## 📝 Deployment & Bitbucket
- Push your code to Bitbucket for team collaboration (see project instructions)
- For cloud or remote deployment, see the deployment section in this README or ask your admin

---

## 🆘 Getting Help

```sh
make help
```

---

## 🛠️ Common Issues
- If you run into errors with ChromaDB/Sqlite3, see [ChromaDB troubleshooting](https://docs.trychroma.com/troubleshooting#sqlite)
- For Redis or Ollama issues, ensure the services are running locally

---

## 📄 License
MIT (or your org's license)
