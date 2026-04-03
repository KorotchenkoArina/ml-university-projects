
# Science Chatbot with Local LLM and RAG

## Project Description

This project implements a **console chatbot based on a locally running Large Language Model (LLM)** combined with **Retrieval-Augmented Generation (RAG)** for searching information in databases.

The chatbot interacts with the user through commands and performs several tasks:

* answers factual questions using a knowledge base,
* recommends scientific books or films,
* extracts structured tabular data from unstructured text,
* solves logical or textual problems step-by-step.

All processing is performed **locally** using **Ollama**, without external cloud APIs.

---

# Project Goal

The goal of this project is to implement a chatbot that demonstrates the integration of:

* **local LLM models**
* **vector search**
* **structured data extraction**
* **chain-of-thought reasoning**

The chatbot must support four main functions:

1. Answer factual questions based on a database.
2. Provide recommendations based on a database of descriptions.
3. Extract structured tables from user-provided text.
4. Solve complex problems with step-by-step reasoning.

---

# Technologies Used

The project uses the following technologies:

* **Python**
* **Ollama** — for running LLM models locally
* **ChromaDB** — vector database
* **RAG (Retrieval-Augmented Generation)**
* **Pydantic** — structured data validation
* **Requests** — interaction with the Ollama API
* **Embeddings model**: `nomic-embed-text`
* **Language models**: `phi`, `mistral`

---

# Chatbot Capabilities

## 1. Answering Factual Questions

The chatbot can answer factual questions using a **vector search over a knowledge base**.

The database contains short texts about major discoveries in the history of science.

Example entries include:

* Isaac Newton — Law of Universal Gravitation
* Albert Einstein — Theory of Relativity
* Marie Curie — Radioactivity research
* Rosalind Franklin — DNA structure research
* Ada Lovelace — Foundations of programming

### How it works

1. The user's question is converted into an **embedding**.
2. ChromaDB retrieves the **most relevant documents**.
3. The retrieved context is inserted into a **prompt**.
4. The local LLM generates a **short factual answer**.

### Example

User input:

```
/science Who developed the theory of relativity?
```

Bot response:

```
Albert Einstein developed the theory of relativity.
```

---

# 2. Recommendation System

The chatbot can recommend **scientific media** such as books, documentaries, or films.

The database contains descriptions of popular science works.

Example items include:

* *A Brief History of Time* — Stephen Hawking
* *Cosmos* — Carl Sagan
* *The Selfish Gene* — Richard Dawkins
* *Hidden Figures* — a film about NASA mathematicians

### How it works

1. The user's request is embedded into a vector.
2. ChromaDB retrieves the **most similar descriptions**.
3. The LLM analyzes the retrieved materials.
4. The model selects **one recommendation** and explains why it fits the request.

### Example

User input:

```
/rec_media I want something about cosmology
```

Bot response:

```
I recommend "A Brief History of Time" (1988).
The book explains the origin of the universe, black holes,
and modern cosmology in an accessible way.
```

---

# 3. Extraction of Structured Tables from Text

The chatbot can analyze a **free-form text** and extract structured information.

In this project the system extracts **scientific institutions** mentioned in the text.

The extracted information includes:

* institution name
* field of science
* location

The chatbot then returns the results as a **Markdown table**.

### Example

Input text:

```
The CERN research center in Geneva studies particle physics.
The Science Museum in London focuses on the history of science.
```

Output table:

| Institution    | Field of science   | Location               |
| -------------- | ------------------ | ---------------------- |
| CERN           | Particle Physics   | Geneva, Switzerland    |
| Science Museum | History of Science | London, United Kingdom |

### Implementation details

This task uses:

* **LLM structured output**
* **JSON schema**
* **Pydantic validation**

The model produces JSON that is validated before converting it into a Markdown table.

---

# 4. Step-by-Step Problem Solving

The chatbot can solve logical or textual problems and explain the reasoning process step by step.

This is implemented using **chain-of-thought prompting**.

### Example

User input:

```
/solve Vasily has a cousin Mikhail who is 20 years older.
How much younger is Vasily than the only son of his mother?
```

Bot response:

```
Step 1. The problem asks how much younger Vasily is than the only son of his mother.

Step 2. Vasily himself is the son of his mother.

Step 3. The problem states that she has only one son.

Step 4. Therefore the only son of his mother is Vasily.

Result: 0 years difference.
```

---

# Architecture Overview

The chatbot follows the **RAG architecture**.

Workflow:

```
User Input
     ↓
Embedding generation
     ↓
Vector search in ChromaDB
     ↓
Relevant context retrieval
     ↓
Prompt construction
     ↓
Local LLM generation
     ↓
Final response
```

---

# Database Structure

The project uses two vector collections.

## 1. Science History Database

Contains information about scientific discoveries.

Each entry includes:

* title
* scientist
* description
* themes

This database is used for **factual question answering**.

---

## 2. Science Media Database

Contains descriptions of science-related media.

Each entry includes:

* title
* year
* category
* description

This database is used for **recommendations**.

---

# Chatbot Commands

The chatbot supports the following commands:

```
/science <question>
```

Ask a factual question from the knowledge base.

```
/rec_media <request>
```

Get a recommendation for a science book, documentary, or film.

```
/table <text>
```

Extract scientific institutions from a text and output them as a table.

```
/solve <problem>
```

Solve a logical or textual problem step by step.

```
/quit
```

Exit the chatbot.

---

# Running the Project

1. Install dependencies:

```
pip install chromadb requests pydantic pandas prettytable
```

2. Install Ollama:

[https://ollama.com](https://ollama.com)

3. Download required models:

```
ollama pull phi
ollama pull mistral
ollama pull nomic-embed-text
```

4. Start Ollama:

```
ollama serve
```

5. Run the chatbot:

```
python chatbot.py
```

---

# Possible Improvements

The project can be extended in several directions:

* expanding the scientific knowledge database
* adding persistent vector storage
* implementing a web interface
* improving prompt engineering
* adding hybrid search (vector + keyword)
* supporting multiple languages

---

# Conclusion

This project demonstrates how a **local LLM chatbot** can be combined with **RAG and vector databases** to perform multiple tasks:

* knowledge retrieval,
* recommendation,
* structured information extraction,
* and reasoning.

The system runs entirely **locally**, making it suitable for experiments, research, and educational purposes.
