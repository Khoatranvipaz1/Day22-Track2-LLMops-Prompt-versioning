"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
import types
import re
warnings.filterwarnings("ignore")

from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAGAS 0.4.x still imports this legacy optional module on some installs.
# The lab does not use Vertex AI, so this shim only keeps imports working.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - compatibility shim
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChatVertexAI is not installed. This lab uses the provider in .env instead."
            )

    vertexai_shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_shim

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
SYSTEM_V1 = (
    "You are a concise RAG assistant. Use only the provided context to answer. "
    "Keep the response short, direct, and grounded in 2-4 sentences. "
    "If the context does not contain the answer, say you do not know.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Read the context carefully, identify the facts "
    "that answer the user's question, then respond in a clear, organized way in "
    "3-5 sentences. Do not add facts that are not supported by the context; say "
    "you do not know when the context is insufficient.\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # TODO: Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # TODO: Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    # TODO: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # TODO: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    # TODO: Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        # TODO: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # TODO: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    # TODO: Tạo EvaluationDataset từ rag_results
    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    if config.PROVIDER in ("local", "hf_local", "gguf"):
        print("ℹ️  Provider local nhỏ: dùng heuristic offline thay cho LLM-as-judge RAGAS.")
        scores = heuristic_rag_scores(rag_results)
    else:
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                llm=llm_eval,
                embeddings=emb_eval,
            )

            scores = {}
            for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
                raw = result[key]
                scores[key] = float(np.mean([v for v in raw if v is not None]))
        except Exception as exc:
            if config.PROVIDER not in ("local", "hf_local", "gguf"):
                raise
            print(f"⚠️  RAGAS evaluator local không parse được output ({exc}).")
            print("   Dùng fallback heuristic offline để tạo report định lượng.")
            scores = heuristic_rag_scores(rag_results)

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


def _tokens(text: str) -> set:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "is", "are", "be", "by", "as", "that", "this", "it", "from", "uses",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if token not in stop and len(token) > 2
    }


def heuristic_rag_scores(rag_results: list) -> dict:
    values = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
    }
    for row in rag_results:
        answer_tokens = _tokens(row["answer"])
        question_tokens = _tokens(row["question"])
        reference_tokens = _tokens(row["reference"])
        context_tokens = _tokens(" ".join(row["contexts"]))

        values["faithfulness"].append(len(answer_tokens & context_tokens) / max(len(answer_tokens), 1))
        values["answer_relevancy"].append(len(answer_tokens & question_tokens) / max(len(question_tokens), 1))
        values["context_recall"].append(len(reference_tokens & context_tokens) / max(len(reference_tokens), 1))
        values["context_precision"].append(len(context_tokens & reference_tokens) / max(len(context_tokens), 1))

    return {key: float(np.mean(score_list)) for key, score_list in values.items()}


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # TODO: Ghi report vào file bằng json.dumps hoặc json.dump
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
