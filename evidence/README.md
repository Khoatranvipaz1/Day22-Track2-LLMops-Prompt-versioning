# Day 22 Lab Report — LangSmith + Prompt Versioning

**Sinh viên:** Trần Văn Khoa — 2A202600827

**LangSmith Project:** https://smith.langchain.com/o/e7534c61-391c-476c-a5b5-18147cafee07/projects (project: `day22-lab`)

---

## Môi trường chạy

- Provider: `openrouter` (model: `openai/gpt-4o-mini` via OpenRouter API)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector store: FAISS
- Framework: LangChain LCEL + LangSmith tracing

---

## Tóm tắt các bước đã hoàn thành

1. **RAG Pipeline (Bước 1):** Tải knowledge base, chia chunks với FAISS, xây dựng RAG chain theo cấu trúc LCEL (`retriever → prompt → LLM → parser`). Trang trí hàm query bằng `@traceable` và chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith dashboard.

2. **Prompt Hub & A/B Routing (Bước 2):** Soạn 2 system prompt khác nhau về ngữ nghĩa (V1: ngắn gọn 2-4 câu; V2: có cấu trúc chuyên sâu 3-5 câu), push lên LangSmith Prompt Hub, pull về khi chạy. Định tuyến tất định bằng MD5 hash của `request_id` — cùng `request_id` luôn cho cùng phiên bản.

3. **RAGAS Evaluation (Bước 3):** Chạy toàn bộ 50 cặp QA qua cả 2 prompt version, xây dựng `EvaluationDataset` với `SingleTurnSample`, tính đủ 4 chỉ số RAGAS.

4. **Guardrails AI (Bước 4):** Triển khai `PIIDetector` (regex cho email, phone, SSN, credit card) và `JSONFormatter` (sửa markdown fences, single quotes, trailing commas). Cả 2 dùng `on_fail=OnFailAction.FIX` truyền vào constructor validator.

---

## Kết quả RAGAS — So sánh V1 vs V2

Model đánh giá: `openai/gpt-4o-mini` qua OpenRouter API

| Chỉ số | Prompt V1 (ngắn gọn) | Prompt V2 (có cấu trúc) | Phiên bản tốt hơn |
|---|---:|---:|:---:|
| faithfulness | 0.7590 | **0.8665** | V2 ✅ |
| answer_relevancy | **0.6521** | 0.6255 | V1 |
| context_recall | 0.7000 | **0.7200** | V2 |
| context_precision | 0.5550 | **0.5583** | V2 |

**Faithfulness đạt mục tiêu ≥ 0.8:** V2 = 0.8665 ✅

---

## Phân tích V1 vs V2

**V2 có faithfulness cao hơn (0.8665 vs 0.7590)** vì system prompt V2 yêu cầu model "đọc kỹ context, xác định facts liên quan" trước khi trả lời. Hướng dẫn rõ ràng này khiến model bám sát tài liệu gốc hơn, giảm thiểu hallucination.

**V1 có answer_relevancy cao hơn (0.6521 vs 0.6255)** vì câu trả lời ngắn hơn (2-4 câu) nên ít bị lạc chủ đề hơn. V2 đôi khi viết quá nhiều chi tiết phụ, làm giảm độ tập trung vào câu hỏi gốc.

**Kết luận:** V2 phù hợp hơn cho ứng dụng RAG yêu cầu độ chính xác cao (faithfulness), còn V1 phù hợp khi cần câu trả lời ngắn gọn và đúng trọng tâm.

---

## Danh sách file bằng chứng

| File | Nội dung |
|------|----------|
| `01_langsmith_rag_pipeline_log.txt` | Log RAG pipeline chạy 50 câu hỏi |
| `01_langsmith_traces.png` | Ảnh chụp LangSmith dashboard ≥ 50 traces |
| `02_ab_routing_log.txt` | Log A/B routing 50 câu với nhãn v1/v2 |
| `02_prompt_hub.png` | Ảnh chụp Prompt Hub hiển thị 2 phiên bản |
| `03_ragas_scores.png` | Bảng so sánh điểm RAGAS V1 vs V2 |
| `03_ragas_report.json` | Báo cáo JSON điểm RAGAS (OpenRouter run) |
| `04_pii_demo_log.txt` | Log demo PII detection & redaction |
| `04_json_demo_log.txt` | Log demo JSON formatter & repair |

---

## Ghi chú về LangSmith

Code hỗ trợ đầy đủ LangSmith tracing và Prompt Hub thông qua `config.py`. Để tạo ảnh `01_langsmith_traces.png` và `02_prompt_hub.png`, cần chạy Bước 1 và Bước 2 trong môi trường có `LANGSMITH_API_KEY` hợp lệ và kết nối internet tới `smith.langchain.com`.
