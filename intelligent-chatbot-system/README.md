# Chatbot with Local LLM, Classifiers and RAG

## Project Description

This project implements an **interactive console chatbot** based on a **locally running Large Language Model (LLM)** combined with several auxiliary systems:

* **content moderation (moral classifier)**
* **automatic function classification**
* **language detection**
* **RAG-based information retrieval**
* **text processing tools**
* **database management**

The chatbot is capable of automatically determining the user's intent and executing the appropriate function without explicit commands.

The system is designed as a **multi-component architecture** combining rule-based classifiers, vector search, and generative language models.

All computations are performed **locally** using **Ollama**, without relying on external cloud APIs.

---

# Project Goals

The main goal of the project is to demonstrate how a chatbot can integrate:

* **local LLM models**
* **RAG (Retrieval-Augmented Generation)**
* **content safety filtering**
* **language detection**
* **automatic intent classification**
* **text analysis tools**

The chatbot provides a unified interface for interacting with multiple NLP tasks.

---

# Key Features

The chatbot includes the following core capabilities.

## 1. Moral Safety Filter

Before processing a request, the chatbot runs a **moral classifier** that detects potentially dangerous or unethical content.

The classifier blocks requests related to:

* weapons
* explosives
* drugs
* violence
* illegal activities
* instructions for dangerous actions

Example blocked queries:

* instructions for building weapons
* drug production recipes
* violent instructions
* hacking instructions

If unsafe content is detected, the chatbot refuses to process the request and explains the reason.

---

# 2. Automatic Function Classification

The chatbot automatically determines **what the user wants to do**.

The system analyzes the request and selects one of the following functions:

* information search (RAG)
* table extraction
* text summarization
* tag generation
* problem solving
* recommendations
* database management

This allows the user to interact naturally without remembering specific commands.

Example:

User input:

```
Explain quantum mechanics
```

Detected function:

```
rag_search
```

---

# 3. Multilingual Support

The chatbot automatically detects the language of the user's request.

Supported languages:

* Russian
* English
* German
* French

The system uses character patterns, common words, and alphabet detection to identify the language.

The chatbot then responds in the same language whenever possible.

---

# 4. RAG-Based Knowledge Search

The chatbot includes a **Retrieval-Augmented Generation system** for searching information in internal databases.

Workflow:

1. The user asks a question.
2. The query is used to search documents stored in a vector database.
3. Relevant documents are retrieved.
4. The chatbot extracts the most relevant information.
5. A concise answer is returned.

Example query:

```
What are black holes?
```

Example response:

```
Black Hole: A black hole is a region of space where gravity is so strong
that nothing, not even light, can escape.
```

---

# 5. User Databases

The system allows users to create and manage **custom knowledge bases**.

Users can:

* create databases
* select active databases
* add documents
* search stored information

Each document is automatically indexed and becomes searchable through the RAG system.

Example commands:

```
/create_db my_articles
/select_db science
/add_doc
```

---

# 6. Table Extraction

The chatbot can convert **unstructured text into structured tables**.

It analyzes the text and generates a **Markdown table** containing organized information.

Example:

Input:

```
Create a table from this text about scientific discoveries.
```

Output:

| Discovery            | Scientist       | Field   |
| -------------------- | --------------- | ------- |
| Theory of Relativity | Albert Einstein | Physics |

This feature is useful for organizing information from raw text.

---

# 7. Text Summarization

The chatbot can produce **short summaries of long texts**.

The system:

1. extracts key sentences
2. uses the LLM to refine the summary
3. returns a concise explanation

Example:

Input:

```
Summarize this text about artificial intelligence.
```

Output:

```
Artificial intelligence is a field of computer science focused on creating
systems capable of performing tasks that normally require human intelligence.
```

---

# 8. Tag and Keyword Generation

The chatbot can analyze text and generate:

* keywords
* key phrases
* thematic tags

The process combines:

* statistical text analysis
* keyword frequency
* LLM-based semantic analysis

Example output:

```
Keywords:
artificial intelligence, machine learning, neural networks

Key phrases:
deep learning models, data training

Tags:
AI, machine learning, neural networks, data science
```

---

# 9. Step-by-Step Problem Solving

The chatbot can solve logical and mathematical problems using **step-by-step reasoning**.

Example query:

```
Solve: x^2 - 5x + 6 = 0
```

Example response:

```
Step 1: Factor the equation.
Step 2: (x - 2)(x - 3) = 0
Step 3: Solve each equation.

Answer:
x = 2 or x = 3
```

---

# System Architecture

The chatbot consists of several independent modules:

```
User Input
    ↓
Language Detection
    ↓
Moral Safety Filter
    ↓
Function Classification
    ↓
Task Processing
    ↓
Response Generation (LLM or internal logic)
```

Main components:

* **MoralClassifier** – content safety filtering
* **FunctionClassifier** – intent detection
* **LanguageClassifier** – language detection
* **DatabaseManager** – user database management
* **SimpleSearchRAG** – document retrieval
* **TextProcessor** – summarization and keyword extraction
* **ChatBot** – main orchestration system

---

# Demonstration Databases

The chatbot automatically creates several demo databases during initialization.

Example datasets include topics from:

### Science

* black holes
* quantum mechanics
* DNA and genetics

### Literature

* *Crime and Punishment* by Fyodor Dostoevsky

### History

* World War II events

These datasets are used for demonstration and testing.

---

# Installation

Install required Python libraries:

```
pip install chromadb requests pydantic numpy
```

---

# Installing Ollama

Download Ollama:

```
https://ollama.com
```

Pull the required model:

```
ollama pull mistral
```

Start Ollama:

```
ollama serve
```

---

# Running the Chatbot

Run the program:

```
python chatbot.py
```

After launch, the chatbot will initialize databases and wait for user input.

---

# Available Commands

The chatbot supports several system commands.

```
/help
```

Show help information.

```
/lang
```

Change the interface language.

```
/list_db
```

Show all available databases.

```
/create_db <name>
```

Create a new database.

```
/select_db <name>
```

Select an active database.

```
/add_doc
```

Add a new document to the current database.

```
/quit
```

Exit the program.

---

# Example Queries

Users can interact with the chatbot using natural language.

Examples:

```
What are black holes?
```

```
Explain quantum mechanics
```

```
Summarize this text about artificial intelligence
```

```
Generate tags for this article
```

```
Create a table from this text
```

```
Solve the equation x^2 - 5x + 6 = 0
```

---

# Possible Improvements

Potential future extensions include:

* using ML-based classifiers instead of keyword rules
* improving language detection accuracy
* persistent vector databases
* web interface (Flask or FastAPI)
* advanced RAG pipelines
* hybrid search (vector + keyword)
* support for additional languages

---

# Conclusion

This project demonstrates how to build a **modular AI chatbot architecture** that combines:

* local language models
* safety filtering
* automatic intent detection
* RAG-based information retrieval
* text analysis tools
* multilingual support

The chatbot runs **entirely locally**, making it suitable for experimentation, research, and educational purposes.