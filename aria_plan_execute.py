# aria_memory.py — Lab 3.2: Memory-Augmented Agent
# Programme: Agentic AI Development for Innovation Team — KPMG
# ─────────────────────────────────────────────────────────────────────────────

# ── IMPORTS ──────────────────────────────────────────────
import json, os, operator
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
# ── CREDENTIALS ───────────────────────────────────────────
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0.2,
)


# ── KNOWLEDGE BASE ─────────────────────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "aria_knowledge_base.json")

with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)["documents"]

print(f"Loaded {len(KB)} documents from knowledge base.")

# ── TOOL 1: SEARCH ─────────────────────────────────────────────────────────────
@tool
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
        match_count = sum(1 for kw in keywords if kw in searchable)
        if match_count >= max(1, len(keywords) // 2):
            results.append(
                f"[{doc['document_id']}] {doc['title']} ({doc['date']})\n"
                f"Type: {doc['type']}\n"
                f"Excerpt: {doc['content'][:300]}..."
            )

    if not results:
        return "No documents found matching that query."
    return "\n\n".join(results)


# ── TOOL 2: GAP CHECK ─────────────────────────────────────────────────────────
REQUIRED_EVIDENCE = {
    "contract_modification" : ["DOC-001"],
    "revenue_schedule"      : ["DOC-002", "DOC-009"],
    "management_assessment" : ["DOC-007"],
    "controller_approval"   : ["DOC-011"],
    "board_awareness"       : ["DOC-003"],
    "auditor_correspondence": ["DOC-005"],
    "accounting_policy"     : ["DOC-012"],
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
# ── TOOL 3: DRAFT REPORT (new in Lab 3.3) ─────────────────────────────────────
@tool
def draft_report(evidence_summary: str) -> str:
    """
    Draft a structured audit findings report from a provided evidence summary.
    Use this as the FINAL step after evidence has been gathered and gaps checked.
    Input: a plain-English summary of the evidence found and gaps identified.
    Returns: a formatted audit findings report with sections for findings,
    gaps, and recommended actions.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an audit report writer for KPMG. "
         "Given an evidence summary, produce a concise structured report with three sections: "
         "1. KEY FINDINGS — what the evidence shows, citing document IDs. "
         "2. EVIDENCE GAPS — what is missing and why it matters. "
         "3. RECOMMENDED ACTIONS — what the engagement team should do next. "
         "Be precise. Use bullet points within each section."),
        ("human", "{evidence_summary}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"evidence_summary": evidence_summary})
# ── PLAN SCHEMA ───────────────────────────────────────────
# The planner LLM will produce a structured Plan object — not free text.
# Pydantic enforces the shape: a list of strings called 'steps'.
class Plan(BaseModel):
    """An ordered execution plan for the audit evidence task."""
    steps: List[str] = Field(
        description="Ordered list of steps to complete the audit evidence task. "
                    "Each step should be a single, concrete action."
    )


# ── AGENT STATE ───────────────────────────────────────────
# This is the shared state object that flows through every node in the graph.
# Every node reads from it and writes back to it.
class PlanExecuteState(TypedDict):
    input: str                              # The original task from the user
    plan: List[str]                         # The upfront plan produced by the planner
    past_steps: Annotated[List, operator.add]  # Steps completed so far (append-only)
    response: str                           # The final consolidated report
# ── NODE 1: PLANNER ───────────────────────────────────────
# Receives the task and produces a complete upfront plan.
# Uses structured output to force the LLM to return a Plan object.
def planner_node(state: PlanExecuteState) -> dict:
    """Produce an ordered plan for the audit evidence task."""
    structured_llm = llm.with_structured_output(Plan)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an audit planning assistant for KPMG. "
         "Given an audit task, produce a clear, ordered step-by-step plan. "
         "Each step must be a single concrete action. "
         "The plan must follow this sequence: "
         "1. Search for relevant evidence documents. "
         "2. Analyse the evidence and check for gaps. "
         "3. Draft a structured findings report. "
         "Return exactly 3 steps."),
        ("human", "{task}"),
    ])

    chain = prompt | structured_llm
    plan_obj = chain.invoke({"task": state["input"]})

    print(f"\n[PLANNER] Produced {len(plan_obj.steps)}-step plan:")
    for i, step in enumerate(plan_obj.steps, 1):
        print(f"  Step {i}: {step}")

    return {"plan": plan_obj.steps}


# ── NODE 2: EXECUTOR ──────────────────────────────────────
# Executes one step at a time from the plan.
# Looks at how many steps have been completed (past_steps)
# and executes the next one in the plan list.
def executor_node(state: PlanExecuteState) -> dict:
    """Execute the next step in the plan using the appropriate tool."""
    plan = state["plan"]
    past = state["past_steps"]
    current_step = plan[len(past)]      # Next step = plan[number of completed steps]

    print(f"\n[EXECUTOR] Executing step {len(past) + 1}/{len(plan)}: {current_step}")

    # Build a prompt that tells the LLM which step to execute and what tools to use
    messages = [
        SystemMessage(content=(
            "You are ARIA, an audit research assistant. "
            "Execute the given step using the available tools. "
            "Available tools: search_knowledge_base, check_evidence_gaps, draft_report. "
            "Use exactly the right tool for the step. Return only the tool result."
        )),
        HumanMessage(content=(
            f"Execute this step: {current_step}\n\n"
            f"Context from previous steps:\n"
            + "\n".join([f"- {s}: {r[:200]}" for s, r in past])
            if past else f"Execute this step: {current_step}"
        )),
    ]

    # Bind the tools to the LLM so it can call them
    llm_with_tools = llm.bind_tools([search_knowledge_base, check_evidence_gaps, draft_report])
    response = llm_with_tools.invoke(messages)

    # If the LLM chose to call a tool, execute it
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find and invoke the correct tool
        tool_map = {
            "search_knowledge_base": search_knowledge_base,
            "check_evidence_gaps": check_evidence_gaps,
            "draft_report": draft_report,
        }
        tool_fn = tool_map.get(tool_name)
        if tool_fn:
            result = tool_fn.invoke(tool_args)
            print(f"[EXECUTOR] Tool called: {tool_name}")
            print(f"[EXECUTOR] Result preview: {str(result)[:200]}...")
        else:
            result = f"Unknown tool: {tool_name}"
    else:
        # LLM responded directly without a tool call
        result = response.content
        print(f"[EXECUTOR] LLM direct response: {result[:200]}...")

    return {"past_steps": [(current_step, result)]}


# ── NODE 3: FINALISER ─────────────────────────────────────
# Called after all plan steps are complete.
# Consolidates everything in past_steps into a final response.
def finaliser_node(state: PlanExecuteState) -> dict:
    """Consolidate all completed steps into a final structured report."""
    print(f"\n[FINALISER] Consolidating {len(state['past_steps'])} completed steps")

    steps_summary = "\n\n".join([
        f"STEP: {step}\nRESULT: {result}"
        for step, result in state["past_steps"]
    ])

    final_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an audit report consolidator for KPMG. "
         "Given the results of all completed audit steps, produce a final "
         "consolidated summary. Include: what was found, what is missing, "
         "and what action is required. Cite document IDs for every finding."),
        ("human", f"Task: {state['input']}\n\nCompleted steps:\n{steps_summary}"),
    ])

    chain = final_prompt | llm | StrOutputParser()
    final_response = chain.invoke({})

    return {"response": final_response}
# ── ROUTING FUNCTION ──────────────────────────────────────
# Called after the planner and after each executor run.
# Decides: is there another step to execute, or are we done?
def should_continue(state: PlanExecuteState) -> str:
    """Route to executor if steps remain, or to finaliser if plan is complete."""
    if len(state["past_steps"]) >= len(state["plan"]):
        return "finalise"
    return "execute"


# ── GRAPH ASSEMBLY ────────────────────────────────────────
# Wire the nodes together into a state graph.
workflow = StateGraph(PlanExecuteState)

# Register nodes
workflow.add_node("planner",   planner_node)
workflow.add_node("executor",  executor_node)
workflow.add_node("finaliser", finaliser_node)

# Entry point: always start with the planner
workflow.add_edge(START, "planner")

# After planning: route to executor or finaliser
workflow.add_conditional_edges(
    "planner",
    should_continue,
    {"execute": "executor", "finalise": "finaliser"},
)

# After each execution: route back to executor (loop) or to finaliser
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {"execute": "executor", "finalise": "finaliser"},
)

# Finaliser always ends the graph
workflow.add_edge("finaliser", END)

# Compile into a runnable
agent = workflow.compile()
print("Plan-and-Execute agent compiled.")
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




# ── MAIN ──────────────────────────────────────────────────────────────────────
# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    TASK = (
        "Conduct a revenue recognition evidence review for Meridian Software FY2024. "
        "Gather all evidence related to contract modifications affecting revenue timing, "
        "assess completeness against audit requirements, and draft a structured "
        "findings report with identified gaps and recommended actions."
    )

    print("\n" + "=" * 60)
    print("ARIA Plan-and-Execute Agent")
    print("FY2024 Revenue Recognition Evidence Review")
    print("=" * 60)

    # Initial state — plan and past_steps start empty
    initial_state = {
        "input": TASK,
        "plan": [],
        "past_steps": [],
        "response": "",
    }

    result = agent.invoke(initial_state, config={"recursion_limit": 20})

    # Print the plan that was produced
    print("\n" + "=" * 60)
    print("UPFRONT PLAN (produced before any execution)")
    print("=" * 60)
    for i, step in enumerate(result["plan"], 1):
        print(f"Step {i}: {step}")

    # Print execution summary
    print("\n" + "=" * 60)
    print(f"EXECUTION SUMMARY ({len(result['past_steps'])} steps completed)")
    print("=" * 60)
    for i, (step, result_text) in enumerate(result["past_steps"], 1):
        print(f"\n[Step {i}] {step}")
        print(f"Result: {result_text[:300]}...")

    # Print final consolidated report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["response"])

    # Print comparison table vs ReAct
    print("\n" + "=" * 60)
    print("COMPARISON: Plan-and-Execute vs ReAct (Lab 3.1)")
    print("=" * 60)
    print(f"Plan-and-Execute: {len(result['plan'])} steps planned upfront")
    print(f"Plan-and-Execute: {len(result['past_steps'])} steps executed")
    print(f"Plan-and-Execute: plan was fixed before execution began")
    print(f"ReAct:            no upfront plan — decides next action after each observation")