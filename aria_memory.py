# ── IMPORTS ──────────────────────────────────────────────
import json, os, sqlite3
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# ── CREDENTIALS ───────────────────────────────────────────
load_dotenv()

# ── LLM (same as Lab 3.1) ─────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0.2,
)

# ── EMBEDDINGS (new for Lab 3.2) ───────────────────────────
# Used to convert text into vectors for semantic memory storage
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
    model=os.getenv('AZURE_OPENAI_EMBEDDING_MODEL'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)

# ── KNOWLEDGE BASE ─────────────────────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "aria_knowledge_base.json") #opening json file and reading the files

with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)["documents"] # in the json file looked for documents key i.e line 5

print(f"Loaded {len(KB)} documents from knowledge base.") #Loaded 12 documents from knowledge base.

# ── TOOL 1: SEARCH ─────────────────────────────────────────────────────────────
# """ dot string, the AI use to read
@tool #with @tool the agent will consider the below code
def search_knowledge_base(query: str) -> str:
    """
    Search the Meridian Software FY2024 audit evidence repository.
    Use this tool to locate documents related to contract modifications,
    revenue recognition assessments, management sign-offs, board minutes,
    auditor correspondence, and policy documents.
    Input: a plain-English search query describing what evidence you need.
    Returns: a list of matching documents with their ID, title, date, and
    a content excerpt. Returns 'No documents found' if nothing matches.
    Do not call this tool more than twice.
    """
    query_lower = query.lower()
    results = []

    for doc in KB:
        searchable = (
            doc["title"].lower() + " " +
            doc["content"].lower() + " " +
            doc["type"].lower()
        )
        keywords = query_lower.split()
        if any(kw in searchable for kw in keywords):
            results.append(
                f"[{doc['document_id']}] {doc['title']} ({doc['date']})\n"
                f"Type: {doc['type']}\n"
                f"Excerpt: {doc['content'][:300]}..."
            )

    if not results:
        return "No documents found matching that query."
    return "\n\n".join(results)

# ── TOOL 2: GAP CHECK ─────────────────────────────────────────────────────────
# Evidence requirements for a complete revenue recognition assessment
# Added the document to have the agent understand and read through these accurate document. To reduce the hallucination.
# This is one of the rule
REQUIRED_EVIDENCE = {
    "contract_modification" : ["DOC-001"],   # amendment document
    "revenue_schedule"      : ["DOC-002", "DOC-009"],  # Q1 and Q2 schedules
    "management_assessment" : ["DOC-007"],   # accounting memo
    "controller_approval"   : ["DOC-011"],   # email sign-off thread
    "board_awareness"       : ["DOC-003"],   # board minutes
    "auditor_correspondence": ["DOC-005"],   # external auditor letter
    "accounting_policy"     : ["DOC-012"],   # policy manual extract
}

@tool
def check_evidence_gaps(found_doc_ids: str) -> str:
    """
    Check whether a set of gathered document IDs covers all required
    evidence categories for a complete FY2024 revenue recognition assessment.
    Use this tool AFTER searching the knowledge base, once you have a list
    of relevant documents. Do NOT use this as the first step.
    Input: a comma-separated string of document IDs, e.g. 'DOC-001,DOC-007'.
    Returns: a structured gap report showing which evidence categories are
    covered, which are missing, and a completeness percentage.
    """
    gathered = {d.strip().upper() for d in found_doc_ids.split(",")}
    report = []
    covered = 0

    for category, required_docs in REQUIRED_EVIDENCE.items():
        found = [d for d in required_docs if d in gathered]
        missing = [d for d in required_docs if d not in gathered]
        status = "COVERED" if found else "MISSING"
        if found:
            covered += 1
        report.append(
            f"{status}: {category} "
            f"(requires {required_docs}, found {found if found else 'nothing'})"
        )

    pct = int((covered / len(REQUIRED_EVIDENCE)) * 100)
    report.insert(
        0,
        f"EVIDENCE COMPLETENESS: {pct}% ({covered}/{len(REQUIRED_EVIDENCE)} categories covered)"
    )
    return "\n".join(report)

# ── SEMANTIC MEMORY STORE ─────────────────────────────────
# Chroma stores embeddings on disk so findings persist across sessions.
# Each stored entry is a text summary + metadata (session_id, doc_ids found).
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "aria_chroma_db")

semantic_memory = Chroma(
    collection_name="aria_findings",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR, #store the findings in the persist directory, next run it will be picked up from this store
)

print(f"Semantic memory store: {semantic_memory._collection.count()} entries loaded.")

def store_finding(session_id: str, summary: str, doc_ids: list) -> None:
    """Store a session finding into semantic memory for future retrieval."""
    doc = Document(
        page_content=summary,
        metadata={
            "session_id": session_id,
            "doc_ids": ",".join(doc_ids),
        }
    )
    semantic_memory.add_documents([doc])
    print(f"Stored finding for session {session_id} in semantic memory.")


def retrieve_relevant_findings(query: str, k: int = 3) -> str:
    """Retrieve past findings relevant to a query from semantic memory."""
    results = semantic_memory.similarity_search(query, k=k)#list of document objects
    if not results:
        return "No relevant past findings in memory."
    parts = []
    for r in results:
        parts.append(
            f"[Session: {r.metadata.get('session_id', 'unknown')}]\n"
            f"Docs found: {r.metadata.get('doc_ids', '')}\n"
            f"Summary: {r.page_content}"
        )
    return "\n\n---\n\n".join(parts)

# ── CONVERSATION MEMORY (SQLite) ───────────────────────────
# SQLite persists the full message history across Python sessions.
# Each thread_id is a separate conversation. Same thread_id = agent
# remembers everything from previous runs with that ID.
SQLITE_PATH = os.path.join(os.path.dirname(__file__), "aria_memory.db")

conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ARIA, the Audit Research Intelligence Assistant for KPMG. "
    "You have persistent memory — you remember what you found in previous "
    "sessions and can build on prior findings rather than starting from scratch. "
    "Your task is to gather and assess audit evidence for the Meridian Software "
    "FY2024 revenue recognition engagement. "
    "Step 1: use search_knowledge_base to locate relevant documents. "
    "Step 2: use check_evidence_gaps with the document IDs you found. "
    "Step 3: produce a structured summary with document references and gaps. "
    "Never assert a finding without a document ID. "
    "Do not call search_knowledge_base more than twice."
)

# ── TOOLS LIST ────────────────────────────────────────────────────────────────
tools = [search_knowledge_base, check_evidence_gaps]

# ── AGENT WITH MEMORY ─────────────────────────────────────────────────────────
# checkpointer= wires SQLite into the agent loop.
# Every message in every session is saved automatically.
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# ── SESSION RUNNER ───────────────────────────────────────
def run_session(thread_id: str, query: str, label: str) -> str:
    """Run one query session and return the final answer."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }

    print(f"\n{'='*60}")
    print(f"{label} | Thread: {thread_id}")
    print(f"{'='*60}")
    print(f"Query: {query}\n")

    result = agent.invoke({"messages": [("human", query)]}, config=config)

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"ACTION: {msg.tool_calls[0]['name']} | "
                  f"INPUT: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"OBSERVATION: {msg.content[:200]}...")

    final = result["messages"][-1].content
    print(f"\nFINAL ANSWER:\n{final}")
    return final
# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # SESSION 1: First engagement day
    answer_1 = run_session(
        thread_id="engagement-meridian-2024",
        query=(
            "ARIA, I need all evidence related to contract modifications that "
            "affected revenue timing in FY2024. Which contracts were modified, "
            "when, what changed, and has management assessed the revenue "
            "recognition impact? Summarise findings and flag any gaps."
        ),
        label="SESSION 1 — Initial evidence gathering"
    )

    # Store Session 1 finding in semantic memory
    store_finding(
        session_id="engagement-meridian-2024-day1",
        summary=answer_1,
        doc_ids=["DOC-001", "DOC-002", "DOC-003", "DOC-007"]
    )

    # SESSION 2: Follow-up — same thread, agent remembers Session 1
    answer_2 = run_session(
        thread_id="engagement-meridian-2024",
        query=(
            "Based on what you found earlier, has the Controller sign-off "
            "for the Northbridge Holdings modification been resolved? "
            "What is still outstanding?"
        ),
        label="SESSION 2 — Follow-up (same thread, agent remembers)"
    )

    # SESSION 3: New thread — agent starts fresh
    answer_3 = run_session(
        thread_id="engagement-meridian-2024-reviewer",
        query=(
            "What do you know about the Northbridge Holdings contract modification?"
        ),
        label="SESSION 3 — New thread (no prior conversation memory)"
    )

    # Demonstrate semantic memory retrieval
    print(f"\n{'='*60}")
    print("SEMANTIC MEMORY RETRIEVAL")
    print(f"{'='*60}")
    print("Query: 'controller approval revenue recognition gap'")
    past = retrieve_relevant_findings("controller approval revenue recognition gap")
    print(past)

    # Show conversation state
    print(f"\n{'='*60}")
    print("CONVERSATION STATE — Thread: engagement-meridian-2024")
    print(f"{'='*60}")
    config = {"configurable": {"thread_id": "engagement-meridian-2024"}}
    state = agent.get_state(config)
    msgs = state.values.get("messages", [])
    print(f"Total messages stored in this thread: {len(msgs)}")
    print(f"Conversation memory: {SQLITE_PATH}")
    print(f"Semantic memory entries: {semantic_memory._collection.count()}")
    
