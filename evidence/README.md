# Day 22 Lab Report

## Runtime

- Provider used: `hf_local`
- Local model: `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`
- Model file: `models/smollm2-360m-instruct-gguf/smollm2-360m-instruct-q8_0.gguf`
- Model size: about 386 MB
- Runtime: `llama-cpp-python`

## Completed Work

1. RAG pipeline was implemented with FAISS, LCEL prompt chaining, local GGUF LLM, and 50 QA runs.
2. Prompt V1 and V2 were implemented with deterministic MD5 request routing.
3. Evaluation ran all 50 QA pairs through both prompt versions.
4. Guardrails validators were implemented and demonstrated:
   - PII redaction for email, phone, SSN, and credit card patterns.
   - JSON repair for markdown fences, single quotes, trailing commas, and invalid JSON fallback.

## Evaluation Results

### OpenRouter API Run

Model: `openai/gpt-4o-mini` via OpenRouter.

| Metric | V1 | V2 | Winner |
|---|---:|---:|---|
| faithfulness | 0.7590 | 0.8665 | V2 |
| answer_relevancy | 0.6521 | 0.6255 | V1 |
| context_recall | 0.7000 | 0.7200 | V2 |
| context_precision | 0.5550 | 0.5583 | V2 |

Best faithfulness was `0.8665`, so the OpenRouter API run met the target `faithfulness >= 0.8`.

### Local GGUF Run

| Metric | V1 | V2 | Winner |
|---|---:|---:|---|
| faithfulness | 0.7208 | 0.6479 | V1 |
| answer_relevancy | 0.6415 | 0.6619 | V2 |
| context_recall | 0.7476 | 0.7476 | Tie |
| context_precision | 0.1240 | 0.1240 | Tie |

Best faithfulness was `0.7208`, below the `0.8` target. This run used a very small 360M local model and an offline heuristic fallback for scoring.

## Evidence Files

- `01_langsmith_rag_pipeline_log.txt`: local RAG run over 50 questions.
- `02_ab_routing_log.txt`: deterministic A/B routing over 50 questions.
- `03_ragas_scores_log.txt`: V1/V2 evaluation run and score table.
- `03_ragas_scores.png`: rendered score table.
- `03_ragas_report.json`: JSON score report copied from `data/ragas_report.json`.
- `03_ragas_scores_openrouter_log.txt`: OpenRouter API RAGAS run log.
- `03_ragas_scores_openrouter.png`: rendered OpenRouter score table.
- `03_ragas_report_openrouter.json`: OpenRouter JSON score report.
- `04_pii_demo_log.txt`: PII validator demo.
- `04_json_demo_log.txt`: JSON formatter demo.

## LangSmith Limitation

The code supports LangSmith tracing and Prompt Hub, and the LangSmith API key/project variables were wired in `config.py`. However, this execution environment blocked external export of prompts, contexts, and traces to hosted LangSmith. Because of that, `01_langsmith_traces.png` and `02_prompt_hub.png` were not produced here. To complete those screenshots, run Step 1 and Step 2 in an environment where sending traces/prompts to LangSmith is allowed.
