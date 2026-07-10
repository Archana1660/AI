# ── IMPORTS ──────────────────────────────────────────────
import json, os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

#load env and LLM
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0.2,
)
# Phase 2 knowledge base loader
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

    try:
    # Deliberate failure injection
        if "contract" in query.lower():
            raise ValueError("Simulated index failure for contract queries")

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
                    f"Except: {doc['content'][:300]}..."
                )

        if not results:
            return "No documents found matching that query."

        return "\n\n".join(results)

    except Exception as e:
        return f"Tool error: {str(e)}. Retry with a different query."
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

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ARIA, the Audit Research Intelligence Assistant for KPMG. "
    "Your task is to gather and assess audit evidence for the Meridian Software "
    "FY2024 revenue recognition engagement. "
    "Step 1: use search_knowledge_base to locate relevant documents. "
    "Step 2: use check_evidence_gaps with the document IDs you found. "
    "Step 3: produce a structured summary with document references and gaps. "
    "Never assert a finding without a document ID. " #this is a guardrail
    "Do not call search_knowledge_base more than twice."  #this is a guardrail
)

# ── TOOLS LIST ────────────────────────────────────────────────────────────────
tools = [search_knowledge_base, check_evidence_gaps]

# ── AGENT ─────────────────────────────────────────────────────────────────────
# create_react_agent returns a CompiledStateGraph
# No AgentExecutor needed — the loop is built into the graph
agent = create_agent( 
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,   # plain string — no hub.pull() needed
)

# Phase 1 connection test
# ===== Check the connection =====
# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
#this goes in the chat interface (user input)
    QUERY = (
        "ARIA, I need all evidence related to contract modifications that affected "
        "revenue timing in FY2024. Specifically: which contracts were modified, "
        "when, what the modification changed, and whether management has assessed "
        "the revenue recognition impact. Please summarise what you find and "
        "flag any gaps in the evidence."
    )

    print("\n" + "=" * 60)
    print("ARIA ReAct Agent — FY2024 Revenue Recognition Review")
    print("=" * 60 + "\n")

    result = agent.invoke(
        {"messages": [("human", QUERY)]},
        config={"recursion_limit": 10}, #max limit step for the conversation is 10th step whether final answer is achieved or not
    )

    # Print the full message trace 
    print("\n--- Full message trace ---")
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"\nACTION: {msg.tool_calls[0]['name']} | INPUT: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"OBSERVATION: {msg.content[:200]}...")
        else:
            print(f"\nMESSAGE: {msg.content[:300]}")

    # Final answer
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["messages"][-1].content)
    print(f"\nTotal messages in trace: {len(result['messages'])}")

    # Structured trace inspection (Phase 4 Task C)
    print("\n--- Structured trace ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc = msg.tool_calls[0]
            print(f"Step {i}: AIMessage → tool_call: {tc['name']} | args: {tc['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"Step {i}: ToolMessage → {msg.name} | result: {str(msg.content)[:120]}...")
        else:
            print(f"Step {i}: {msg_type} → {str(msg.content)[:80]}...")
