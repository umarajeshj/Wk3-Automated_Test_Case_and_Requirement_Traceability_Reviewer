# Requirement & Test Traceability Reviewer (QA AI Pipeline)

Multi-agent parallel AI pipeline with Human-in-the-Loop (HITL) escalation for reviewing Product Requirement Documents (PRDs) against test suites. Built with **LangGraph**, **Ollama (`gemma4:e4b`)**, and **Streamlit**.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

---

## Quick Start

1. **Pull Ollama Models & Serve:**
   ```bash
   ollama serve
   ollama pull gemma4:e4b