# LLM-based Synthetic Event Log Generator

This project is a command-line framework for generating synthetic process-mining event logs in XES format from natural-language requirements. It uses IBM watsonx models for intent detection, clarification, and Python script generation, with optional GraphRAG knowledge retrieval and BPMN process-model input.

## What It Does

- Collects event-log requirements through an interactive CLI.
- Detects generation intents such as process structure, path distribution, throughput time, and resource utilization.
- Asks clarification questions when requirements are incomplete.
- Optionally retrieves process-mining knowledge from the bundled GraphRAG index.
- Generates and executes a Python script that creates an XES event log.
- Validates the generated XES file and creates basic throughput-time and variant reports.

## Current Project Layout

```text
.
├── main.py                 # CLI entry point
├── tools.py                # LLM wrapper helpers
├── requirements.txt        # Runtime dependencies
├── pyproject.toml          # Project metadata
├── .env.example            # Local environment template
├── data/
│   ├── input/              # Optional user-provided BPMN input
│   └── models/             # Sample BPMN/text process models
├── event_log/              # Generated run folders are written here
├── generation/             # Prompt construction and generation loop
├── intent_detection/       # Intent detection logic
├── memory/                 # SQLite-backed conversation memory
├── eval/                   # Runtime XES validation and report helpers
├── graph_rag/
│   ├── input/              # Source documents for the knowledge base
│   ├── output/             # Lightweight prebuilt GraphRAG parquet tables
│   ├── prompts/            # GraphRAG prompt templates
│   └── settings.yaml       # GraphRAG configuration
└── prompt/                 # Main framework prompt templates
```

There are no test datasets, notebooks, or historical run archives in the cleaned project. Runtime files such as `history.db`, `event_log/run_*`, GraphRAG caches, logs, and local `.env` files are ignored by git.

## Requirements

- Python 3.11 or newer
- IBM watsonx credentials
- Network access to IBM watsonx for real generation runs
- `GEMINI_API_KEY` if you want GraphRAG retrieval to call Gemini-backed models

## Setup

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```text
WATSONX_API_KEY=
WATSONX_PROJECT_ID=
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_GENERAL_MODEL_ID=openai/gpt-oss-120b
WATSONX_JUDGE_MODEL_ID=meta-llama/llama-3-3-70b-instruct
GEMINI_API_KEY=
```

The model IDs can be changed if your watsonx project uses different deployed or available models.

## Running

Start the interactive CLI:

```bash
python main.py
```

Example request:

```text
Generate a synthetic event log for an order handling process with order received, payment check, packaging, shipping, and delivery confirmation.
```

After you have provided enough requirements, you can ask the framework to generate:

```text
Start the generation now.
```

Generated outputs are written to:

```text
event_log/run_<timestamp>/
```

Each successful run normally contains:

- `generate_log.py`
- `event_log.xes`
- `report.txt`
- `prompt.txt`

## Optional BPMN Input

Place at most one `.bpmn` file in:

```text
data/input/
```

If this folder is empty, generation uses only the conversation requirements.

## GraphRAG Behavior

The project includes lightweight prebuilt GraphRAG parquet tables in `graph_rag/output/`. Larger generated GraphRAG state is not tracked:

- `graph_rag/cache/`
- `graph_rag/logs/`
- `graph_rag/update_output/`
- `graph_rag/output/lancedb/`

If GraphRAG retrieval fails because a generated vector store or external model call is unavailable, the CLI skips retrieval and continues with empty knowledge context.

## Security

- Never commit `.env`.
- Never commit API keys, generated logs, local databases, or GraphRAG caches.
- `graph_rag/settings.yaml` uses `${GEMINI_API_KEY}` placeholders instead of real credentials.
- Rotate any credentials that were previously stored in local files before publishing the repository.
