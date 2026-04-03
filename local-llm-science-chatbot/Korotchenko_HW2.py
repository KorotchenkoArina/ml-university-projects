import json
import requests
import chromadb
from pydantic import BaseModel, ValidationError
from typing import List
import re
import prettytable as pt
import pandas as pd
from typing import Any, Dict


# ----------------------------------------------------------------------
# 1. PYDANTIC MODELS (FOR NEW DATA)
# ----------------------------------------------------------------------
class OllamaEmbeddingFunction:
    def __init__(self, model="nomic-embed-text", host="http://localhost:11434"):
        self.model = model
        self.host = host

    # Called when adding documents
    def __call__(self, input):
        return self._get_embeddings(input)

    # Called when searching / querying
    def embed_query(self, input):
        return self._get_embeddings(input)

    # Helper – do not duplicate code
    def _get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
            embeddings.append(response.json()["embedding"])
        return embeddings

    # Chroma calls this method to check for name conflicts
    def name(self):
        return f"ollama-{self.model}"


class InstitutionRow(BaseModel):
    """Одна строка таблицы."""
    institution: str          # название учреждения
    field_of_science: str    # область(и) науки (можно перечислять через запятую)
    location: str            # город, страна (или регион)


class InstitutionTableAnswer(BaseModel):
    """Ответ LLM – список строк таблицы."""
    rows: list[InstitutionRow]


class ScientificTopic(BaseModel):
    title: str
    scientist: str
    description: str
    key_concepts: List[str]


class ScienceMedia(BaseModel):
    title: str
    year: int
    category: List[str]
    description: str

SYSTEM_PROMPT_INSTITUTIONS = """
You are a data‑extraction specialist.

Your task is to read the supplied free‑form text and extract **all mentions of scientific
institutions** (museums, research centres, laboratories, universities, observatories, etc.).
For each institution you must output **exactly three columns** in this order:

| Institution | Field of science | Location |
|-------------|------------------|----------|

* **Institution** – the official name as it appears in the source text.  
* **Field of science** – one or several scientific areas, separated by commas.  
* **Location** – city and country (or region).

**Return ONLY the markdown table** – no prose, no JSON, no code fences, no extra lines.
If no institutions are found, return the table header **only** (an empty table).

**Example of a correct answer**:

| Institution | Field of science | Location |
|-------------|------------------|----------|
| Science Museum (London) | History of Science, Technology | London, United Kingdom |
| CERN | Particle Physics, Accelerator Science | Geneva, Switzerland |
"""


# ----------------------------------------------------------------------
# 2. DATABASE PREPARATION
# ----------------------------------------------------------------------
# Database 1 – History of Science & Technology
science_topics_base = [
    {
        "title": "Law of Universal Gravitation",
        "scientist": "Isaac Newton",
        "content": """Isaac Newton formulated the law of universal gravitation,
        which states that every mass in the universe attracts every other mass.
        This discovery became a cornerstone of classical mechanics.""",
        "key_question": "Who formulated the law of universal gravitation?",
        "answer": "Isaac Newton",
        "themes": ["gravity", "classical mechanics", "physics"],
    },
    {
        "title": "Theory of Relativity",
        "scientist": "Albert Einstein",
        "content": """Albert Einstein developed the special and general theories of relativity,
        fundamentally changing our understanding of space, time, and gravity.""",
        "key_question": "Who developed the theory of relativity?",
        "answer": "Albert Einstein",
        "themes": ["space", "time", "gravity", "physics"],
    },
    {
        "title": "Radioactivity Research",
        "scientist": "Marie Curie",
        "content": """Marie Curie pioneered research on radioactivity and discovered
        two new chemical elements: polonium and radium.
        She was the first woman to win a Nobel Prize and the only person
        to win Nobel Prizes in two different scientific fields.""",
        "key_question": "Who pioneered research on radioactivity?",
        "answer": "Marie Curie",
        "themes": ["radioactivity", "chemistry", "physics"],
    },
    {
        "title": "DNA Structure Discovery",
        "scientist": "Rosalind Franklin",
        "content": """Rosalind Franklin made critical contributions to the discovery
        of the molecular structure of DNA through her work on X‑ray diffraction.
        Her data played a key role in understanding the double helix.""",
        "key_question": "Who contributed crucial data to the discovery of DNA structure?",
        "answer": "Rosalind Franklin",
        "themes": ["genetics", "biology", "DNA"],
    },
    {
        "title": "Computer Programming Foundations",
        "scientist": "Ada Lovelace",
        "content": """Ada Lovelace is considered the first computer programmer.
        She wrote the first algorithm intended to be processed by a machine
        and foresaw the potential of computers beyond numerical calculations.""",
        "key_question": "Who is considered the first computer programmer?",
        "answer": "Ada Lovelace",
        "themes": ["computer science", "algorithms", "programming"],
    },
]

# Database 2 – Popular Science Books & Movies
science_media_base = [
    {
        "title": "A Brief History of Time",
        "year": 1988,
        "category": ["Popular Science", "Cosmology"],
        "description": """Stephen Hawking’s book explores the origin of the universe,
        the nature of time, black holes, and modern cosmology in an accessible way.""",
    },
    {
        "title": "Cosmos",
        "year": 1980,
        "category": ["Documentary", "Astronomy"],
        "description": """Carl Sagan’s documentary series explores the universe,
        scientific discovery, and humanity’s place in the cosmos.""",
    },
    {
        "title": "The Selfish Gene",
        "year": 1976,
        "category": ["Biology", "Evolution"],
        "description": """Richard Dawkins presents evolution from the gene‑centered
        perspective, offering a new way to understand natural selection.""",
    },
    {
        "title": "Hidden Figures",
        "year": 2016,
        "category": ["Film", "History of Science"],
        "description": """A film telling the story of African‑American women mathematicians
        who played a crucial role in NASA’s early space missions.""",
    },
]


# ----------------------------------------------------------------------
# 3. CHROMADB INITIALISATION
# ----------------------------------------------------------------------
embedding_function = OllamaEmbeddingFunction()

client = chromadb.Client()

science_collection = client.get_or_create_collection(
    name="science_history", embedding_function=embedding_function
)
for i, topic in enumerate(science_topics_base):
    science_collection.add(
        ids=[f"science_{i}"],
        documents=[topic.get("content", "")],
        metadatas=[{"title": topic["title"], "scientist": topic["scientist"]}],
    )

print("[DEBUG] Documents in science_collection:", science_collection.count())
print(
    "[DEBUG] Test query 'Newton':",
    science_collection.query(query_texts=["Newton"], n_results=2),
)

media_collection = client.get_or_create_collection(
    name="science_recommendations", embedding_function=embedding_function
)
for i, media in enumerate(science_media_base):
    media_collection.add(
        ids=[f"media_{i}"],
        documents=[media["description"]],
        metadatas=[
            {
                "title": media["title"],
                "year": media["year"],
                "category": ", ".join(media["category"]),
            }
        ],
    )


# ----------------------------------------------------------------------
# 4. INTERACTION WITH OLLAMA
# ----------------------------------------------------------------------
def get_ollama_response(
    prompt: str, model: str = "phi", ollama_host: str = "http://localhost:11434"
):
    print("[DEBUG] Entering get_ollama_response")
    print("[DEBUG] Model:", model)
    print("[DEBUG] Prompt length:", len(prompt))
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.4,
            },
            timeout=1000,
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama is not reachable. Start it with: ollama serve"
    except requests.exceptions.ReadTimeout:
        return "[ERROR] Ollama is responding too slowly. Try a smaller model."
    except Exception as e:
        return f"[ERROR] {str(e)}"


# ----------------------------------------------------------------------
# 5. CORE CHATBOT FUNCTIONS
# ----------------------------------------------------------------------
def task_1_factual_question(question: str) -> str:
    """
    Task 1 – answer factual questions using the scientific facts DB.
    Works via Ollama + ChromaDB embeddings.
    """

    # 1️⃣ Retrieve results from Chroma
    results = science_collection.query(query_texts=[question], n_results=2)

    # Keep only the top‑1 most relevant document
    if results and results["documents"] and results["documents"][0]:
        top_doc = results["documents"][0][0].strip()
    else:
        top_doc = None

    if not top_doc:
        return "Sorry, the information for your question was not found in the database."

    # Build a clean prompt for the LLM
    prompt = f"""
You are an expert in the history of science. Using the context below,
answer **only with facts**, briefly and to the point.
If the question concerns a person, name them immediately.
If the answer is not present in the context, say "Information not found".

Context:
{top_doc}

Question:
{question}

Answer:
"""

    # 4️⃣ Send the prompt to Ollama
    response = get_ollama_response(
        prompt=prompt, model="phi", ollama_host="http://localhost:11434"
    )

    # 5️⃣ Debug output (can be removed later)
    print("[DEBUG] Prompt sent to Ollama:\n", prompt)
    print("[DEBUG] Ollama response:\n", response)

    # 6️⃣ Return the answer
    return response if response else "Sorry, Ollama did not return a response."


def task_2_recommendation(
    preference: str,
    model: str = "phi",
    ollama_host: str = "http://localhost:11434",
    n_results: int = 3,
) -> str:
    """
    Returns a single recommendation from `media_collection` that best matches
    the user's request. If nothing is found, a friendly message is returned.
    """
    # Query the vector store
    results = media_collection.query(query_texts=[preference], n_results=n_results)

    # Debug output (optional)
    print("[DEBUG] media_collection.query →", results)

    # Guard against empty results
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    if not docs or not docs[0]:
        return "No media materials were found for your request."

    # Build the *context* – concatenated descriptions
    context = "\n".join(docs[0])

    # Build a readable metadata block for the prompt
    metadata_str = "\n".join(
        [", ".join([f"{k}: {v}" for k, v in meta.items()]) for meta in metas[0]]
    )

    # Assemble the prompt
    prompt = f"""You are a recommendation system for science and education.

User request:
{preference}

Found materials (descriptions):
{context}

Material metadata:
{metadata_str}

Your task is to choose **one** material that best satisfies the request and provide a concise but convincing recommendation
(title, year, and why it is relevant)."""

    # Send to Ollama
    answer = get_ollama_response(
        prompt=prompt,
        model=model,
        ollama_host=ollama_host,
    )

    # Handle possible empty reply
    if not answer:
        return "Error: LLM did not return a response."

    return answer

def rows_to_markdown_table(rows: list[InstitutionRow]) -> str:
    if not rows:
        return "| Institution | Field of science | Location |\n|-------------|------------------|----------|\n*No institutions found*"
        
    table = "| Institution | Field of science | Location |\n"
    table += "|-------------|------------------|----------|\n"
        
    for row in rows:
        institution = str(row.institution).replace("|", "\\|")
        field = str(row.field_of_science).replace("|", "\\|")
        location = str(row.location).replace("|", "\\|")
        
        table += f"| {institution} | {field} | {location} |\n"
    
    return table

def task_3_extract_table(text: str) -> str:
    import requests
    import json
    from pydantic import ValidationError
    
    SYSTEM_PROMPT_INSTITUTIONS_JSON = """
    You are a data-extraction specialist. Your task is to read the supplied text and extract 
    all mentions of scientific institutions (museums, research centres, laboratories, 
    universities, observatories, etc.). Return the data in JSON format matching the required schema.
    
    For each institution provide:
    1. institution - official name as it appears in the text
    2. field_of_science - scientific area(s), separated by commas if multiple
    3. location - city and country (or region)
    
    If no institutions are found, return an empty list.
    
    Return ONLY valid JSON that matches the schema.
    """
    
    # Prepare the prompt for structured extraction
    user_prompt = f"""
    Extract all scientific institutions from the following text:
    
    {text}
    
    Return the data as a JSON object with a "rows" field containing a list of institution objects.
    Each object must have: institution, field_of_science, and location fields.
    """
    
    json_schema = InstitutionTableAnswer.model_json_schema()
    
    try:
        print("[DEBUG] мы зашли в try")
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_INSTITUTIONS_JSON},
                    {"role": "user", "content": user_prompt}
                ],
                "format": json_schema,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=2000
        )
        
        if response.status_code != 200:
            return f"Error: Ollama API returned status {response.status_code}"
        
        response_data = response.json()
        content = response_data.get("message", {}).get("content", "")
        
        if not content:
            return "Error: Empty response from Ollama"
        
        try:
            parsed_data = json.loads(content)
            validated_data = InstitutionTableAnswer(**parsed_data)
            
            return rows_to_markdown_table(validated_data.rows)
            
        except json.JSONDecodeError as e:
            return f"Error parsing JSON from Ollama: {e}\nRaw response: {content}"
        except ValidationError as e:
            return f"Error validating data: {e}\nRaw response: content"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running on http://localhost:11434"
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def task_4_step_by_step_solution(problem: str) -> str:
    """Solve a problem step‑by‑step."""
    prompt = f"""Solve the problem step by step.

Problem:
{problem}

Response format:
Step 1...
Step 2...
Result:"""

    return get_ollama_response(prompt = prompt, model="mistral", ollama_host="http://localhost:11434")


# ----------------------------------------------------------------------
# 6. CHATBOT INTERFACE
# ----------------------------------------------------------------------
MENU_TEXT = """
/science <question>      – ask a factual science question
/rec_media <request>    – get a media (book/film) recommendation
/table <text>           – extract a table of institutions from text
/solve <task>           – step‑by‑step solution of a problem
/quit                    – exit the program
"""

def print_menu() -> None:
    print("\n=== Available commands ===")
    print(MENU_TEXT.strip())


def handle_command(user_input: str) -> str:
    """Dispatches the command string to the appropriate task."""
    if user_input.startswith("/science "):
        return task_1_factual_question(user_input[5:])
    elif user_input.startswith("/rec_media "):
        return task_2_recommendation(user_input[10:])
    elif user_input.startswith("/table "):
        return task_3_extract_table(user_input[7:])
    elif user_input.startswith("/solve "):
        return task_4_step_by_step_solution(user_input[7:])
    else:
        return "Unknown command. Please use one of the listed commands."


def chatbot_interface():
    print("CHATBOT WITH RAG AND LOCAL LLM (SCIENCE HISTORY)")
    print_menu()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "/quit":
            print("[INFO] Exiting chatbot – goodbye!")
            break

        answer = handle_command(user_input)
        print("\nAnswer:")
        print(answer)

        # Show the menu again so the user always knows what's available
        print_menu()


# ----------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Initialising chatbot...")
    print(f"[INFO] Loaded {len(science_topics_base)} scientific topics")
    print(f"[INFO] Loaded {len(science_media_base)} media items")
    print("[INFO] Make sure Ollama is running")
    chatbot_interface()