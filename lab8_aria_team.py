"""
Lab 8 -- ARIA Audit Team
A four-agent AutoGen GroupChat (researcher -> analyst -> reviewer-> compliance_officer) running
on the Module 5 tool layer. The memo write stays OUTSIDE the chat entirely:
deterministic code, gated on reviewer approval.

Module 8 | Agentic AI Development for Innovation Teams
Engagement context: Meridian Software Ltd FY2024
"""

import asyncio
import json
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
# from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console #only for visibility not part of agentic design
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv
from pathlib import Path
# Same tool layer as Lab 5 -- unchanged, just re-orchestrated.
from aria_tools import DATA_STORE, FAILURE_CONFIG, execute_tool
# ------------------------------------------------------------------
# Model client
# ------------------------------------------------------------------
# gpt-5.4 is not a model name AutoGen recognises, so model_info is
# REQUIRED -- without it the client raises at construction time.
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-5.4",
    model="gpt-5.4",
    api_version="2025-01-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    model_info={
        "vision": False,# doesn't support images
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown", #not to consume specific data but be general
    },
)
# ====
# Captures the analysis handle so post-chat code can find the flagged
# data without parsing it out of chat messages.
LAST_ANALYSIS = {"handle": None}

# load_dotenv(Path(__file__).resolve().parent.parent / "module5" / ".env")
load_dotenv()  # load .env from current working directory


# ------------------------------------------------------------------
# Tool wrappers -- AutoGen builds schemas from type hints + docstrings
# ------------------------------------------------------------------
def fetch_ledger(section: str) -> str:
    """Fetch one section ('revenue' or 'expenses') of the Meridian Software Ltd
    FY2024 engagement ledger. Returns JSON containing a data handle and record
    count -- never raw records."""
    return json.dumps(execute_tool("fetch_engagement_data", {"section": section}))


def flag_variances(handle: str, threshold_pct: float = 15.0) -> str:
    """Flag ledger accounts whose absolute year-on-year variance meets or exceeds
    threshold_pct. Requires a data handle from fetch_ledger. Returns JSON with a
    new handle and the flagged account names."""
    outcome = execute_tool(
        "analyze_variance", {"handle": handle, "threshold_pct": threshold_pct}
    )
    if outcome["status"] == "ok":
        LAST_ANALYSIS["handle"] = outcome["result"]["handle"]
    return json.dumps(outcome)

# ------------------------------------------------------------------
# The team -- one tool per specialist, none for the reviewer
# ------------------------------------------------------------------
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[fetch_ledger],
    system_message="""You are the RESEARCHER on the Meridian Software Ltd FY2024
audit team (KPMG Deals Advisory).
Your only job: fetch the ledger section named in the task and report the data
handle and record count to the team. Never analyse. Never draft narrative.
If a tool returns status "error" with retryable=false, report the failure in one
sentence and end your message with the single word HALT.""",
)

analyst = AssistantAgent(
    name="analyst",
    model_client=model_client,
    tools=[flag_variances],
    system_message="""You are the ANALYST on the Meridian Software Ltd FY2024
audit team (KPMG Deals Advisory).
Take the researcher's data handle, flag variances at the threshold stated in the
task, then write a 2-4 sentence professional narrative covering ONLY the accounts
named in your tool output. Never invent figures or account names.
If the reviewer requests changes, revise the narrative only -- do not re-run tools
unless the reviewer explicitly questions the data.
If a tool returns status "error" with retryable=false, report the failure in one
sentence and end your message with the single word HALT.""",
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="""You are the REVIEWER -- the quality gate on the Meridian
Software Ltd FY2024 audit team. You have no tools; you cannot fetch or analyse.
Check the analyst's narrative against the tool outputs visible in this chat:
1. Every account mentioned appears in the analyst's tool output -- nothing invented.
2. No figures appear that did not come from tool output.
3. The narrative is 2-4 sentences and professional in tone.
If any check fails, state precisely what must change. If all checks pass, reply
with the single word APPROVED and nothing else.""",
)

compliance_officer = AssistantAgent(
    name="compliance_officer",
    model_client=model_client,
    system_message="""You are the COMPLIANCE OFFICER on the Meridian Software Ltd FY2024 audit team.

Review ONLY the analyst's final narrative.

Do not review tool outputs, reviewer comments, or user instructions.

If no analyst narrative has been produced yet, reply exactly:

WAITING_FOR_ANALYST

Otherwise:
1. Verify every document identifier mentioned in the narrative exists in the ledger or tool outputs.
2. If any document reference cannot be verified, reply:
REJECTED: <reason>
3. If all document references are valid, reply only:
APPROVED""",
)

termination = (
    TextMentionTermination("APPROVED")
    | TextMentionTermination("HALT")
    | MaxMessageTermination(12)
)

team = SelectorGroupChat(
    [researcher, analyst, reviewer, compliance_officer],
    model_client=model_client,
    termination_condition=termination,
)

TASK = (
    "Produce an approved variance narrative for the Meridian Software Ltd FY2024 "
    "revenue ledger. Flag variances of 15% or more."
)


# ------------------------------------------------------------------
# Deterministic write-back -- outside the chat, gated on approval
# ------------------------------------------------------------------
def finalise(result) -> None:
    """No agent holds a write tool. The memo is written by this code, and only
    if the reviewer approved. Module 3 principle, third appearance."""
    transcript = [
        m for m in result.messages if isinstance(getattr(m, "content", None), str)
    ]
    approved = any(
        m.source == "reviewer" and "APPROVED" in m.content for m in transcript
    )
    if not approved:
        print("\n[gate] No reviewer approval in transcript -- memo NOT written.")
        return
    if LAST_ANALYSIS["handle"] is None:
        print("\n[gate] Approval found but no analysis handle -- memo NOT written.")
        return

    narrative = next(
        m.content for m in reversed(transcript) if m.source == "analyst"
    )
    outcome = execute_tool(
        "draft_findings_memo",
        {"handle": LAST_ANALYSIS["handle"], "analyst_summary": narrative},
    )
    print(f"\n[gate] Reviewer approved -- write-back result: {outcome}")
    
async def main() -> None:
    # FAILURE_CONFIG["fetch_engagement_data"] = "timeout"
    # FAILURE_CONFIG["analyze_variance"] = "corrupt"

    result = await Console(team.run_stream(task=TASK))
    finalise(result)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())