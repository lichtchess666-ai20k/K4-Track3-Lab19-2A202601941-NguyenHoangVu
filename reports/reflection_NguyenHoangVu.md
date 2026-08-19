# Reflection — Nguyen Hoang Vu (2A202601941)

## 1. Mapping bài giảng → code

| Module bài giảng | Hàm/cell trong notebook | Kết quả lần chạy |
|---|---|---|
| M1. Preprocessing & Coreference | `standardize_news()`, `near_dedup()`, `build_chunks()`, `run_coref()` | 2118 bài → 2118 chunk |
| M2. NER + RE theo strict schema | `run_extraction()`, `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | 555 triple |
| M3. Entity Resolution | `build_resolution_map()`, `merge_guard()`, class `UF` (union-find) | 907 mention → 898 entity |
| M4. Bulk ingestion `UNWIND` | `bulk_insert_nodes()`, `bulk_insert_edges()` | 898 node / 554 edge trong 1.4s |
| M5. Hybrid retrieval + Evaluation | `retrieve_flat_context()`, `retrieve_graph_context()`, `answer_graph_rag()`, `run_evaluation()` | 25 câu Golden qua LLM Judge |

## 2. Lỗi khó nhất và cách xử lý

- **Schema dataset không cố định:** `pick_col()` theo allowlist bị trượt khi data dump đổi tên cột → thêm fallback chọn cột object có độ dài trung bình lớn nhất và in ra tên cột đã chọn để kiểm chứng bằng mắt.
- **LLM trả JSON kèm rác/markdown fence:** `parse_json_object()` cắt fence rồi lấy đoạn `{...}` ngoài cùng; mỗi batch coref/extract đều bọc try/except để một batch hỏng không giết cả pipeline.
- **Judge (OpenAI/Groq) bị rate limit:** thêm `judge_answer_safe()` có retry + backoff; thất bại cuối cùng vẫn trả điểm 1 kèm rationale `JUDGE_FAILED` để không mất cả lần evaluation đã chạy tốn kém.
- **Seed matching trượt:** khi `match_seeds()` trả về rỗng thì GraphRAG mất hoàn toàn phần graph context và tụt về Flat RAG. Đã thêm fuzzy fallback bằng embedding (ngưỡng 0.66) và ghi `NO_SEED` vào diagnostics để truy vết trong failure analysis.
- **Near-dedup ở quy mô lớn:** bài học rõ nhất là ngưỡng LSH và ngưỡng Jaccard là hai thứ khác nhau — LSH chỉ *sinh candidate* (recall), Jaccard thật mới *quyết định* (precision).

## 3. Kế hoạch áp dụng vào đồ án

**Bài toán của tôi có cần GraphRAG không?** Chỉ cần khi câu hỏi thực tế đòi hỏi **nối nhiều quan hệ qua nhiều tài liệu**
(ví dụ: truy vết chuỗi sở hữu/đầu tư, phân tích ảnh hưởng giữa các thực thể). Nếu 90% câu hỏi chỉ là tra cứu một đoạn văn bản,
Flat RAG rẻ hơn và ít điểm hỏng hơn — số liệu nhóm `factoid` ở trên cho thấy đúng điều đó.

**Thiết kế sơ bộ:**

- **Nodes:** `Company`, `Person`, `Technology`, `Product`, `Event` (đều mang base label `Entity`).
- **Relations:** giữ allowlist nhỏ; mọi cạnh bắt buộc có `source_chunk_id`, `published_date`, `evidence`, `confidence`.
- **Chiến lược dữ liệu:** near-dedup trước khi chunk; NER rẻ tiền lọc chunk trước khi gọi LLM; entity resolution có bảng audit và duyệt tay top-N cặp similarity cao.
- **Retrieval:** router (factoid → vector; multi-hop/cross-doc → hybrid graph) + super-node cap + self-correction hop 2 → hop 3 → vector fallback.
- **Đo lường:** Golden Dataset chia 3 nhóm, chạy lại mỗi lần đổi prompt/threshold; theo dõi cả quality lẫn latency/token để luôn biết cái giá phải trả.
