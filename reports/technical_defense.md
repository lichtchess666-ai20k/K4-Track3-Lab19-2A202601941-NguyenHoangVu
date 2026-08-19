# Thuyết minh Kỹ thuật — Lab 19: Production-Grade GraphRAG vs Flat RAG

**Sinh viên:** Nguyen Hoang Vu — **MSSV:** 2A202601941
**Stack:** Neo4j AuraDB · Groq `gpt-4o-mini` (extract + generate) · `sentence-transformers/all-MiniLM-L6-v2` (embedding) · Judge `openai/gpt-4o-mini`

## 0. Số liệu lần chạy

| Chỉ số | Giá trị |
|---|---|
| Bài viết sau exact dedup | 2118 |
| Bài viết sau near-dedup (MinHash/LSH) | 2118 |
| Chunk index cho Flat RAG | 2118 |
| Chunk gửi LLM trích xuất | 2118 |
| Triple thô trích xuất | 555 |
| Node / Edge trong Neo4j | 898 / 554 |
| Edge thiếu provenance | 0 |
| Câu hỏi Golden | 25 |

---

## 1. Coreference sai ở tình huống nào? Hậu quả?

Prompt coref được thiết kế **conservative**: chỉ phân giải khi tiền ngữ nằm ngay trong cùng chunk, phần còn lại đẩy vào `unresolved_mentions` thay vì đoán.
Lần chạy này: 431/2118 chunk bị sửa text và 148 mention được **giữ nguyên**.

Tình huống sai điển hình trong tin công nghệ là **hai công ty cùng xuất hiện trước đại từ**:

> *"Microsoft announced a partnership with OpenAI. **It** later acquired a robotics startup."*

`It` có thể là Microsoft hoặc OpenAI. Nếu LLM chọn nhầm, pipeline sinh ra cạnh `OpenAI -ACQUIRED-> X` hoàn toàn sai
nhưng **vẫn có đủ provenance** (`source_chunk_id`, `published_date`) nên sanity check không bắt được.
Cạnh sai đó sau này sẽ được BFS kéo vào context và phá cả *faithfulness* lẫn *multi-hop reasoning* — đây là lỗi **im lặng**, nguy hiểm hơn nhiều so với lỗi làm chương trình dừng.

Chunk minh hoạ còn mention chưa phân giải: `8e922bc62b578e73e815::c0000` — `['It']`.

**Cách chặn đang dùng:** (1) giữ nguyên văn bản khi mơ hồ, (2) chunk overlap 40 từ để tiền ngữ ít bị cắt qua chunk, (3) bắt buộc trường `evidence` cho mọi relation nên luôn audit ngược được cạnh nghi ngờ.

---

## 2. Entity threshold bao nhiêu, vì sao?

- Cosine similarity (FAISS `IndexFlatIP`, vector đã normalize): **≥ 0.9**, lấy top-5 láng giềng.
- Lexical guard sau đó: trùng tên sau khi bỏ hậu tố pháp nhân, hoặc `SequenceMatcher ≥ 0.72` (Company) / `≥ 0.84` (Technology).

Lý do chọn 0.90 chứ không phải 0.80: MiniLM cho điểm rất cao với **mọi cặp tên công ty công nghệ** vì chúng cùng miền ngữ nghĩa.
Ở mức 0.80, "Google" và "Microsoft" đã có thể lọt vào candidate. Ngưỡng 0.90 vẫn giữ được recall cho biến thể viết tắt/hậu tố
(*Microsoft Corp* ↔ *Microsoft*) nhưng đẩy phần lớn nhiễu ra ngoài. Quan trọng hơn: embedding chỉ dùng để **sinh candidate**,
quyết định merge cuối cùng luôn thuộc về lexical guard — đó là lý do không cần ép threshold quá chặt.

Kết quả audit: {'MERGE_VECTOR': 7, 'REJECT_GUARD': 5, 'MERGE_MANUAL': 3} trên tổng 15 dòng;
907 mention → 898 canonical entity.

---

## 3. Candidate nào similarity cao nhưng KHÔNG nên merge?

**Amazon Web Services (AWS)** vs **Amazon Web Services** — similarity 0.929, bị guard chặn với lý do `extra_qualifier_tokens:aws`.

Bốn nhóm bị guard chặn (Challenge B):

1. **Ticker** — `MSFT` vs `Microsoft` chỉ được merge qua `MANUAL_ALIASES`, không bao giờ qua vector: chuỗi 4 chữ cái viết hoa nằm gần rất nhiều thứ trong không gian embedding.
2. **Product chứa tên công ty** — `Apple Watch` vs `Apple`, `Google Cloud` vs `Google`: token bao hàm nhưng token thừa không phải hậu tố pháp nhân → reject. Nếu merge, đồ thị mất hoàn toàn khả năng phân biệt **sản phẩm** với **nhà sản xuất**.
3. **Phiên bản** — `GPT-4` vs `GPT-3`: khác tập chữ số → reject.
4. **Người trùng họ** — `Sam Altman` vs `Steve Altman`: cùng họ, tên riêng khác → reject; chỉ chấp nhận khi tên riêng trùng hoặc là chữ cái viết tắt (`S. Altman`).

Ngược lại `Meta Platforms` vs `Meta` **được** merge vì token thừa (`platforms`) nằm trong danh sách token tổ chức chung.

Bảng audit đầy đủ: `outputs/entity_resolution_audit.csv` — cột `reason` ghi rõ căn cứ của từng quyết định.

---

## 4. Top 3 super-node và degree

| name | type | degree |
|---|---|---|
| Microsoft | Company | 15 |
| Google | Company | 12 |
| L&T Technology Services | Company | 11 |

Chính sách: `SUPER_NODE_DEGREE = 100` → cắt còn `SUPER_NODE_EDGE_CAP = 50` cạnh mới nhất,
trần toàn cục `GLOBAL_EDGE_CAP = 250` cạnh và `MAX_GRAPH_CONTEXT_CHARS = 14000` ký tự.

---

## 5. Vì sao ưu tiên edge mới nhất có thể đúng/sai?

**Đúng khi:** câu hỏi mang tính trạng thái hiện tại (*ai đang lãnh đạo X?*, *X vừa mua lại ai?*). Tin công nghệ thay đổi nhanh, cạnh cũ dễ tạo ra câu trả lời lỗi thời mà model vẫn phát biểu rất tự tin.

**Sai khi:** câu hỏi mang tính lịch sử hoặc cross-doc (*dòng vốn của Meta thay đổi thế nào qua thời gian?*).
Cắt theo `published_date DESC` sẽ **xoá mất dấu vết lịch sử**, tức là làm hỏng đúng nhóm câu hỏi mà GraphRAG lẽ ra thắng.
Rủi ro thứ hai: bài không có ngày xuất bản mang `published_date = ''` nên bị đẩy xuống cuối và gần như không bao giờ được chọn.

**Cải tiến đề xuất:** thay "top-50 mới nhất" bằng **quota lai** — 25 cạnh mới nhất + 15 cạnh confidence cao nhất + 10 cạnh cũ nhất; hoặc xếp hạng cạnh theo độ liên quan ngữ nghĩa giữa `evidence` và câu hỏi.

---

## 6–7. Flat RAG thắng nhóm nào? GraphRAG thắng nhóm nào?

- **cross-doc**: GraphRAG thắng (Flat 3.94 · Graph 4.24)
- **factoid**: Hoà thắng (Flat 5.0 · Graph 5.0)
- **multi-hop**: Flat RAG thắng (Flat 3.58 · Graph 3.53)

Bảng tổng hợp:

| Metric | Flat RAG | GraphRAG | Delta (Graph - Flat) |
|---|---|---|---|
| Comprehensiveness | 3.72 | 3.84 | 0.12 |
| Faithfulness | 4.12 | 4.2 | 0.08 |
| Multi-hop reasoning | 3.72 | 3.84 | 0.12 |
| Latency (s) | 2.511 | 2.508 | -0.004 |
| Token usage | 768.64 | 1376.56 | 607.92 |

Chi tiết theo nhóm: `outputs/graphrag_vs_flatrag_summary.csv`. Hai ca lỗi phân tích sâu: `reports/failure_analysis.md`.

Diễn giải: GraphRAG có lợi thế ở multi-hop và cross-doc vì subgraph đã **nối sẵn** các mắt xích nằm rải rác ở nhiều chunk, trong khi vector search phải hy vọng cả chuỗi suy luận cùng rơi vào top-k. Ngược lại ở nhóm factoid, câu trả lời thường nằm gọn trong một đoạn văn — lúc đó graph context chỉ thêm nhiễu và tốn token.

---

## 8. Latency / token trade-off

- Latency trung bình: Flat **2.51s** vs Graph **2.51s** (chênh -0.00s).
- Token trung bình: Flat **769** vs Graph **1377** (chênh +608).

Con số latency ở trên **chỉ tính lần gọi LLM sinh câu trả lời**. Chi phí thật của GraphRAG còn gồm: 1 lần LLM trích seed entity,
N truy vấn Cypher cho BFS (mỗi node 2 query: đo degree + lấy cạnh mới nhất), cộng với vector search.
Đổi lại, graph context là **danh sách cạnh đã nén** nên ngắn hơn nhiều so với 6 chunk văn bản thô — token không tăng tuyến tính theo lượng thông tin.

Kết luận vận hành: dùng **router** — câu factoid đi thẳng Flat RAG (rẻ, nhanh), chỉ câu multi-hop/cross-doc mới kích hoạt GraphRAG.

---

## 9. AI Coding Agent đề xuất gì mà tôi KHÔNG dùng?

1. **Pairwise cosine O(N²) cho near-dedup.** Với 1500 bài thì chạy được, nhưng ở 350MB (hàng trăm nghìn bài) là ~10¹⁰ phép so sánh. Đã thay bằng **MinHash 128 perm + LSH 32 band**, chỉ tính Jaccard thật trên 504 cặp candidate.
2. **`MERGE` từng dòng trong vòng lặp Python.** Mỗi row một round-trip tới AuraDB. Đã thay bằng `UNWIND $rows AS row` theo batch 1000.
3. **Hạ threshold entity resolution xuống 0.75 để "gộp được nhiều hơn".** Tăng recall nhưng tạo false merge im lặng — và false merge là loại lỗi **không sửa ngược được** sau khi đã ghi vào graph.
4. **Bỏ lexical guard cho gọn.** Guard chính là thứ duy nhất chặn `Apple Watch` gộp vào `Apple`.
5. **Dùng `apoc.*` cho traversal.** Không đảm bảo AuraDB free tier có plugin → giữ Cypher thuần để notebook chạy được ở mọi môi trường.

Nguyên tắc kiểm soát Agent: Agent viết code, nhưng **threshold, allowlist schema và chính sách cắt tỉa đều phải có bảng audit và test kiểm chứng** — `entity_resolution_audit.csv`, `test_supernode_policy()`, `test_edge_provenance()`.

---

## 10. Scale 350MB: bottleneck đầu tiên là gì?

Bottleneck **không** phải Neo4j mà là **bước gọi LLM trích xuất NER+RE**: hiện tại 555 triple từ 2118 chunk, mỗi request 4 chunk.
Ngoại suy tuyến tính sang ~350MB (hàng triệu chunk) thì riêng phần extraction đã vượt xa quota và thời gian cho phép.

Thứ tự nút cổ chai và cách xử lý:

1. **LLM extraction (nghiêm trọng nhất):** lọc trước bằng NER rẻ tiền (spaCy) để chỉ gửi chunk *có thực thể công ty/người*; tăng batch; chạy song song nhiều worker với hàng đợi có retry/backoff; cache theo hash của chunk để không trả tiền hai lần.
2. **Embedding + FAISS in-memory:** `IndexFlatIP` là brute-force và giữ toàn bộ vector trong RAM → chuyển sang `IndexIVFFlat`/HNSW hoặc vector store ngoài (Qdrant/pgvector), hoặc dùng vector index ngay trên node Neo4j.
3. **Coreference toàn corpus:** thay LLM bằng model coref chuyên dụng (fastcoref), hoặc chỉ chạy coref cho chunk thực sự có đại từ mơ hồ.
4. **Neo4j write throughput:** batch 1000 vẫn hợp lý; lớn hơn nữa thì dùng `apoc.periodic.iterate` hoặc `neo4j-admin import` offline cho lần nạp đầu tiên.
5. **Super-node ngày càng nặng:** ở quy mô thật, Google/Microsoft sẽ có degree hàng chục nghìn → phải chuyển sang tỉa theo cửa sổ thời gian và dùng community-level summary thay vì đọc cạnh thô.
