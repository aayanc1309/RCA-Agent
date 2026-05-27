# Incident Root Cause Analysis Agent

An AI-powered production-grade tool that performs automated root cause analysis on incident logs, stack traces, and error outputs using **Google Gemini 2.0 Flash Lite**, **LangGraph**, and **Streamlit**.

---

## ✨ Features

- **Multi-language parser** — Python, Java, Node.js, Go, Ruby, generic fallback (pure regex, no LLM)
- **Parallel tool execution** — codebase search + git history scan run concurrently via LangGraph
- **Structured LLM output** — Gemini enforces JSON schema, Pydantic validates every field
- **Rich UI** — dark-mode Streamlit dashboard with confidence bars, fix-step cards, and postmortem download
- **Graceful degradation** — works without a git repo; codebase search skips missing dirs

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and replace `your_key_here` with your real Gemini API key
```

Get a free key at: https://aistudio.google.com/apikey

### 3. Run the Streamlit app

```bash
# From inside the incident_rca_agent/ directory:
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## 📁 Project Structure

```
incident_rca_agent/
├── .env.example              # API key template
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── config.py                 # All tuneable constants
├── llm_client.py             # GeminiClient wrapper (retry, structured output)
├── schemas.py                # Pydantic v2 output models
├── prompts.py                # All prompt strings
├── tools/
│   ├── parse_stack_trace.py  # Regex-based multi-language parser
│   ├── search_codebase.py    # Local filesystem search
│   └── check_git_history.py  # git log / grep integration
├── agent/
│   ├── state.py              # LangGraph AgentState TypedDict
│   ├── nodes.py              # Node functions
│   └── graph.py              # Graph topology + run_agent()
├── app.py                    # Streamlit UI
└── sample_inputs/
    ├── python_error.txt      # Django AttributeError sample
    ├── java_error.txt        # Spring Boot NullPointerException sample
    └── node_error.txt        # Express.js DB connection error sample
```

---

## 🧪 Trying It Out

Paste one of the sample inputs (or drag-and-drop the files) from `sample_inputs/` into the UI and click **Analyse Incident**.

For best codebase-search results, run the app from your **project root**:

```bash
cd /path/to/your/project
streamlit run /path/to/incident_rca_agent/app.py
```

---

## 📝 Notes

| Topic | Detail |
|---|---|
| **Codebase search** | Searches the current working directory; run from your project root for relevant results |
| **Git history** | Requires CWD to be inside a git repo; gracefully no-ops if not |
| **Internet access** | Only the Gemini API call requires internet; all other processing is local |
| **Parallel nodes** | `search_codebase` and `check_git_history` run concurrently via LangGraph fan-out |
| **Retries** | LLM calls retry up to 3× with exponential back-off on API errors |
| **Structured output** | Gemini's JSON mode + Pydantic validation ensures every field is present and typed |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google AI Studio API key |

---

## 🏗 Architecture

```
User input (paste / upload)
        │
        ▼
  node_parse_input  ←── pure regex parser (no LLM)
        │
   ┌────┴────┐  parallel fan-out
   ▼         ▼
node_search  node_check_git
_codebase    (subprocess git)
   │         │
   └────┬────┘  fan-in
        ▼
  node_analyse  ←── Gemini 2.0 Flash Lite (structured JSON)
        │
        ▼
   AgentOutput  →  Streamlit UI
```
