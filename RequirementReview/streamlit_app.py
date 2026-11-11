import operator
import json
import uuid
import datetime
import tempfile
import os
from typing import Annotated, TypedDict, Literal

# ── LangChain / LangGraph ────────────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphInterrupt

# ── Pydantic ─────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

# ── PDF ──────────────────────────────────────────────────────────────
from pypdf import PdfReader

# ── Streamlit ────────────────────────────────────────────────────────
import streamlit as st

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="QA Requirement Reviewer", page_icon="🧪")

# ════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════

class RequirementData(BaseModel):
    feature_name:          str   = Field(description="Name of feature or module")
    total_requirements:    int   = Field(description="Total count of explicit requirements/user stories")
    total_test_cases:      int   = Field(description="Total count of associated test cases provided")
    mapped_requirements:   int   = Field(description="Count of requirements that have at least 1 test case mapped")
    critical_reqs_count:   int   = Field(description="Count of P0/Critical priority requirements")
    unmapped_critical_count: int = Field(description="Count of P0/Critical requirements with zero test coverage")
    negative_tests_count:  int   = Field(description="Count of negative/boundary/error-handling test cases")
    module_priority:       Literal["P0", "P1", "P2"] = Field(description="Overall business priority of this feature")


class PolicyFindings(BaseModel):
    status:           Literal["PASS", "FAIL", "CONDITIONAL"]
    failed_rules:     list[str] = Field(default_factory=list)
    compliance_notes: list[str] = Field(default_factory=list)
    confidence:       float     = Field(ge=0.0, le=1.0)
    reasoning:        str


class LogicFindings(BaseModel):
    clarity_assessment:   str
    edge_case_coverage:   Literal["EXCELLENT", "ADEQUATE", "POOR", "MISSING"]
    testability_health:   Literal["STRONG", "MODERATE", "WEAK"]
    strengths:            list[str] = Field(default_factory=list)
    concerns:             list[str] = Field(default_factory=list)
    confidence:           float     = Field(ge=0.0, le=1.0)
    reasoning:            str


class CoverageFindings(BaseModel):
    coverage_score:        float = Field(ge=0, le=100)
    gap_severity_score:    float = Field(ge=0, le=100)
    composite_risk_score:  float = Field(ge=0, le=100)
    risk_tier:             Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    mitigations:           list[str] = Field(default_factory=list)
    confidence:            float     = Field(ge=0.0, le=1.0)
    reasoning:             str


class PanelDecision(BaseModel):
    decision:            Literal["APPROVED", "NEEDS_REVISION", "REJECTED", "ESCALATE"]
    overall_quality_score: float   = Field(ge=0, le=100)
    requires_escalation: bool
    action_items:        list[str] = Field(default_factory=list)
    critical_gaps:       list[str] = Field(default_factory=list)
    executive_summary:   str
    panel_confidence:    float     = Field(ge=0.0, le=1.0)


# ════════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════════

class SpecialistResult(TypedDict):
    specialist: str
    findings:   dict
    confidence: float


class ReviewState(TypedDict):
    review_id:           str
    document_text:       str
    data:                RequirementData | None
    specialist_results:  Annotated[list[SpecialistResult], operator.add]
    panel_decision:      PanelDecision | None
    requires_escalation: bool
    human_decision:      str | None
    human_notes:         str | None
    final_report:        dict | None


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION + QA POLICY RULES
# ════════════════════════════════════════════════════════════════════

MODEL = "gemma4:e4b"

QA_RULES = {
    "min_total_coverage_pct": 90.0,  # Total test coverage must be >= 90%
    "max_unmapped_p0":         0,     # Zero unmapped critical requirements allowed
    "min_negative_tests_pct": 15.0,  # Negative tests must be >= 15% of total tests
}


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

class CoverageRiskInput(BaseModel):
    total_reqs:  int = Field(description="Total requirement count")
    mapped_reqs: int = Field(description="Mapped requirement count")
    unmapped_p0: int = Field(description="Unmapped P0 critical requirement count")

@tool(args_schema=CoverageRiskInput)
def calc_coverage_gap_risk(total_reqs: int, mapped_reqs: int, unmapped_p0: int) -> str:
    """Calculate requirement coverage risk score (0-100, higher = riskier)."""
    unmapped_total = max(0, total_reqs - mapped_reqs)
    pct_unmapped = (unmapped_total / max(1, total_reqs)) * 100
    
    base_score = pct_unmapped * 0.6
    p0_penalty = min(40.0, unmapped_p0 * 20.0)
    score = round(min(100.0, base_score + p0_penalty), 1)
    
    return json.dumps({
        "coverage_risk_score": score,
        "unmapped_pct": round(pct_unmapped, 1),
        "p0_penalty_applied": p0_penalty
    })


class TestRatioInput(BaseModel):
    total_tests:    int = Field(description="Total test case count")
    negative_tests: int = Field(description="Negative test case count")

@tool(args_schema=TestRatioInput)
def calc_test_distribution_risk(total_tests: int, negative_tests: int) -> str:
    """Calculate risk score based on negative and edge-case test ratios."""
    neg_pct = (negative_tests / max(1, total_tests)) * 100
    risk_score = max(0.0, (20.0 - neg_pct) * 4.0)
    score = round(min(100.0, risk_score), 1)
    return json.dumps({"negative_test_pct": round(neg_pct, 1), "distribution_risk_score": score})

COVERAGE_TOOLS = [calc_coverage_gap_risk, calc_test_distribution_risk]


# ════════════════════════════════════════════════════════════════════
# NODES
# ════════════════════════════════════════════════════════════════════

def node_extract(state: ReviewState) -> dict:
    llm = ChatOllama(model=MODEL, temperature=0, num_ctx=8192)
    structured = llm.with_structured_output(RequirementData)
    data = structured.invoke([
        SystemMessage(content=(
            "You are a Senior QA Automation Architect. Extract feature requirements and test case statistics "
            "precisely from the provided PRD and Test Spec document."
        )),
        HumanMessage(content=f"Extract requirement data from:\n\n{state['document_text'][:6000]}"),
    ])
    return {
        "data":                data,
        "specialist_results":  [],
        "requires_escalation": False,
        "human_decision":      None,
        "human_notes":         None,
        "panel_decision":      None,
        "final_report":        None,
    }


def node_policy_gatekeeper(state: ReviewState) -> dict:
    data = state["data"]
    failed_rules     = []
    compliance_notes = []

    cov_pct = (data.mapped_requirements / max(1, data.total_requirements)) * 100.0
    neg_pct = (data.negative_tests_count / max(1, data.total_test_cases)) * 100.0

    if cov_pct < QA_RULES["min_total_coverage_pct"]:
        failed_rules.append(f"Total coverage {cov_pct:.1f}% below minimum limit of {QA_RULES['min_total_coverage_pct']}%")

    if data.unmapped_critical_count > QA_RULES["max_unmapped_p0"]:
        failed_rules.append(f"Found {data.unmapped_critical_count} unmapped P0 requirements (Max allowed: {QA_RULES['max_unmapped_p0']})")

    if neg_pct < QA_RULES["min_negative_tests_pct"]:
        compliance_notes.append(f"Negative test ratio {neg_pct:.1f}% is below target {QA_RULES['min_negative_tests_pct']}%")

    if data.module_priority == "P0":
        compliance_notes.append("P0 Critical Feature Module — mandatory senior QA sign-off required")

    status = "FAIL" if failed_rules else ("CONDITIONAL" if compliance_notes else "PASS")

    findings = PolicyFindings(
        status=status,
        failed_rules=failed_rules,
        compliance_notes=compliance_notes,
        confidence=1.0,
        reasoning=f"Rule-based policy check against QA standards. Violations: {len(failed_rules)}"
    )
    return {"specialist_results": [{"specialist": "policy", "findings": findings.model_dump(), "confidence": findings.confidence}]}


def node_logic_analyst(state: ReviewState) -> dict:
    llm = ChatOllama(model=MODEL, temperature=0, num_ctx=8192)
    data = state["data"]
    structured = llm.with_structured_output(LogicFindings)
    findings = structured.invoke([
        SystemMessage(content=(
            "You are a Lead QA Engineer. Evaluate requirement clarity, edge-case thoroughness, "
            "and testability of the specifications."
        )),
        HumanMessage(content=f"Requirement Data:\n{json.dumps(data.model_dump(), indent=2)}"),
    ])
    return {"specialist_results": [{"specialist": "logic", "findings": findings.model_dump(), "confidence": findings.confidence}]}


def node_coverage_quantifier(state: ReviewState) -> dict:
    llm = ChatOllama(model=MODEL, temperature=0, num_ctx=8192)
    data = state["data"]
    agent = create_react_agent(llm, COVERAGE_TOOLS, response_format=CoverageFindings)
    result = agent.invoke({"messages": [SystemMessage(content=(
        f"You are a Quantitative QA Metrics Risk Analyst.\n"
        f"Data: {json.dumps(data.model_dump(), indent=2)}\n\n"
        "Step 1: Call calc_coverage_gap_risk.\n"
        "Step 2: Call calc_test_distribution_risk.\n"
        "Step 3: Composite risk = 60% coverage risk + 40% distribution risk."
    ))]})
    findings = result["structured_response"]
    return {"specialist_results": [{"specialist": "coverage", "findings": findings.model_dump(), "confidence": findings.confidence}]}


def node_synthesize(state: ReviewState) -> dict:
    llm = ChatOllama(model=MODEL, temperature=0, num_ctx=8192)
    results = state["specialist_results"]
    data    = state["data"]
    by_type = {r["specialist"]: r["findings"] for r in results}
    avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)

    structured = llm.with_structured_output(PanelDecision)
    decision   = structured.invoke([
        SystemMessage(content=(
            "You are the QA Director. Aggregate specialist reviews and make a sign-off decision.\n"
            "Options: APPROVED | NEEDS_REVISION | REJECTED | ESCALATE\n"
            "Rules for ESCALATE: Unmapped P0 requirements > 0, OR P0 Module Priority, OR panel confidence < 0.65."
        )),
        HumanMessage(content=(
            f"Requirement stats:\n{json.dumps(data.model_dump(), indent=2)}\n\n"
            f"Policy Panel:\n{json.dumps(by_type.get('policy', {}), indent=2)}\n\n"
            f"Logic Panel:\n{json.dumps(by_type.get('logic', {}), indent=2)}\n\n"
            f"Coverage Risk Panel:\n{json.dumps(by_type.get('coverage', {}), indent=2)}\n\n"
            f"Average Panel Confidence: {avg_conf:.0%}"
        )),
    ])

    escalate = (
        decision.requires_escalation
        or decision.decision == "ESCALATE"
        or data.unmapped_critical_count > 0
        or data.module_priority == "P0"
        or avg_conf < 0.65
    )
    return {"panel_decision": decision, "requires_escalation": escalate}


def node_human_review(state: ReviewState) -> dict:
    data = state["data"]
    decision = state["panel_decision"]
    packet = {
        "review_id":           state["review_id"],
        "feature_name":        data.feature_name,
        "priority":            data.module_priority,
        "unmapped_p0_count":   data.unmapped_critical_count,
        "ai_decision":         decision.decision,
        "overall_quality":     decision.overall_quality_score,
        "confidence":          decision.panel_confidence,
        "summary":             decision.executive_summary,
        "critical_gaps":       decision.critical_gaps,
        "action_items":        decision.action_items,
        "prompt":              "Enter: APPROVED | REJECTED | NEEDS_REVISION  (optionally add '| notes')",
    }
    human_input = interrupt(packet)

    if isinstance(human_input, dict):
        h_decision = human_input.get("decision", decision.decision).upper()
        h_notes    = human_input.get("notes", "")
    else:
        parts      = str(human_input).strip().split("|", 1)
        h_decision = parts[0].strip().upper()
        h_notes    = parts[1].strip() if len(parts) > 1 else ""

    return {"human_decision": h_decision, "human_notes": h_notes}


def node_generate_report(state: ReviewState) -> dict:
    data = state["data"]
    decision = state["panel_decision"]
    results  = state["specialist_results"]
    by_type  = {r["specialist"]: r["findings"] for r in results}
    final_decision = state.get("human_decision") or decision.decision

    report = {
        "review_id":   state["review_id"],
        "timestamp":   datetime.datetime.now().isoformat(),
        "extracted_data": data.model_dump(),
        "specialist_panel": {
            "policy":   by_type.get("policy"),
            "logic":    by_type.get("logic"),
            "coverage": by_type.get("coverage"),
        },
        "ai_recommendation": {
            "decision":          decision.decision,
            "overall_quality":   decision.overall_quality_score,
            "confidence":        decision.panel_confidence,
            "critical_gaps":     decision.critical_gaps,
            "action_items":      decision.action_items,
            "executive_summary": decision.executive_summary,
        },
        "final_outcome": {
            "decision":       final_decision,
            "escalated":      state["requires_escalation"],
            "human_override": state.get("human_decision") is not None,
            "human_notes":    state.get("human_notes"),
        },
    }
    return {"final_report": report}


# ════════════════════════════════════════════════════════════════════
# GRAPH
# ════════════════════════════════════════════════════════════════════

def fan_out_to_specialists(state: ReviewState) -> list[Send]:
    return [
        Send("policy_gatekeeper",    state),
        Send("logic_analyst",        state),
        Send("coverage_quantifier",  state),
    ]

def route_post_synthesis(state: ReviewState) -> str:
    return "human_review" if state["requires_escalation"] else "generate_report"

@st.cache_resource
def build_graph():
    checkpointer = MemorySaver()
    builder      = StateGraph(ReviewState)

    builder.add_node("extract_data",          node_extract)
    builder.add_node("policy_gatekeeper",     node_policy_gatekeeper)
    builder.add_node("logic_analyst",         node_logic_analyst)
    builder.add_node("coverage_quantifier",   node_coverage_quantifier)
    builder.add_node("synthesize",            node_synthesize)
    builder.add_node("human_review",          node_human_review)
    builder.add_node("generate_report",       node_generate_report)

    builder.add_edge(START, "extract_data")
    builder.add_conditional_edges(
        "extract_data",
        fan_out_to_specialists,
        ["policy_gatekeeper", "logic_analyst", "coverage_quantifier"],
    )
    builder.add_edge("policy_gatekeeper",    "synthesize")
    builder.add_edge("logic_analyst",        "synthesize")
    builder.add_edge("coverage_quantifier",  "synthesize")
    builder.add_conditional_edges(
        "synthesize",
        route_post_synthesis,
        ["human_review", "generate_report"],
    )
    builder.add_edge("human_review",    "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile(checkpointer=checkpointer), checkpointer


# ════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════════════════════════

# Session state setup
for k, v in [("pipeline_result", None), ("graph_config", None), ("review_id", None), 
             ("escalation_pending", False), ("run_complete", False), ("escalation_packet", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧪 Requirement & Test Traceability Reviewer")

with st.sidebar:
    st.title("⚙️ QA Policy Rules")
    st.write(f"**Model:** `{MODEL}`")
    st.write(f"**Min Coverage Limit:** {QA_RULES['min_total_coverage_pct']}%")
    st.write(f"**Max Unmapped P0:** {QA_RULES['max_unmapped_p0']}")
    st.write(f"**Min Negative Tests:** {QA_RULES['min_negative_tests_pct']}%")

uploaded = st.file_uploader("Upload PRD / Test Suite PDF", type=["pdf"])
if st.button("▶ Run Traceability Pipeline", disabled=(uploaded is None), type="primary"):
    st.session_state.pipeline_result = None
    st.session_state.escalation_pending = False
    st.session_state.run_complete = False

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        doc_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.unlink(tmp_path)

    review_id = f"REV-{uuid.uuid4().hex[:6].upper()}"
    config = {"configurable": {"thread_id": review_id}}
    st.session_state.review_id = review_id
    st.session_state.graph_config = config

    initial: ReviewState = {
        "review_id": review_id, "document_text": doc_text, "data": None,
        "specialist_results": [], "panel_decision": None, "requires_escalation": False,
        "human_decision": None, "human_notes": None, "final_report": None
    }

    graph, _ = build_graph()

    with st.status("Analyzing requirements & test coverage...", expanded=True) as status:
        try:
            for event in graph.stream(initial, config=config, stream_mode="updates"):
                node_name = next(iter(event))
                st.write(f"✔ Completed node: **{node_name}**")

            final_state = graph.get_state(config)
            if final_state.values.get("final_report"):
                st.session_state.pipeline_result = final_state.values
                st.session_state.run_complete = True
                status.update(label="Analysis Complete", state="complete")
            else:
                st.session_state.pipeline_result = final_state.values
                st.session_state.escalation_pending = True
                st.session_state.escalation_packet = final_state.tasks[0].interrupts[0].value
                status.update(label="Paused — QA Sign-off Required", state="running")
        except GraphInterrupt:
            final_state = graph.get_state(config)
            st.session_state.pipeline_result = final_state.values
            st.session_state.escalation_pending = True
            st.session_state.escalation_packet = final_state.tasks[0].interrupts[0].value
            status.update(label="Paused — QA Sign-off Required", state="running")

# Render Output
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    data = res.get("data")
    if data:
        st.subheader("📊 Extracted Requirement Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Feature", data.feature_name)
        c2.metric("Total Reqs", data.total_requirements)
        c3.metric("Mapped Reqs", data.mapped_requirements)
        c4.metric("Unmapped P0", data.unmapped_critical_count)

    if st.session_state.escalation_pending and not st.session_state.run_complete:
        st.warning("⚠️ High Risk or P0 Requirements Detected — Manual QA Sign-off Triggered")
        pkt = st.session_state.escalation_packet
        if pkt:
            st.json(pkt)
        notes = st.text_area("QA Lead Sign-off Notes:")
        
        col_a, col_b = st.columns(2)
        if col_a.button("✅ Approve Sign-off", type="primary"):
            graph, _ = build_graph()
            final = graph.invoke(Command(resume={"decision": "APPROVED", "notes": notes}), config=st.session_state.graph_config)
            st.session_state.pipeline_result = final
            st.session_state.run_complete = True
            st.session_state.escalation_pending = False
            st.rerun()
        if col_b.button("❌ Reject / Request Revision"):
            graph, _ = build_graph()
            final = graph.invoke(Command(resume={"decision": "NEEDS_REVISION", "notes": notes}), config=st.session_state.graph_config)
            st.session_state.pipeline_result = final
            st.session_state.run_complete = True
            st.session_state.escalation_pending = False
            st.rerun()

    if res.get("final_report"):
        st.success("📋 Final Audit Report Available")
        st.json(res["final_report"])