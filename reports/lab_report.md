# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Hoàng Vũ · **MSSV:** 2A202601941
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Cấu hình lần chạy được báo cáo**

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | HackerNoon tech-company-news-data-dump — **5.000 dòng đầu** (đúng phạm vi Golden Dataset của giảng viên) |
| Sau lọc `len ≥ 80` + exact dedup | **2.118 bài** → 2.118 chunk |
| Coreference | Groq `openai/gpt-oss-120b` (`reasoning_effort=low`) |
| NER + RE, sinh câu trả lời | OpenAI `gpt-4o-mini` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (FAISS `IndexFlatIP`) |
| LLM-as-a-Judge | `gpt-4o-mini` |
| Graph | Neo4j AuraDB 5.27 — **898 node / 554 cạnh** |

> **Ghi chú trung thực về môi trường chạy:** kế hoạch ban đầu chạy toàn bộ pipeline trên Groq, nhưng free tier giới hạn **200.000 token/ngày** và riêng bước coreference đã tiêu hết ~193k. Vì vậy từ bước trích xuất trở đi chuyển sang `gpt-4o-mini`. Điều này **không phá vỡ tính công bằng của benchmark**: Flat RAG và GraphRAG dùng **chung một generator và chung một judge**, chỉ khác nhau ở tầng retrieval.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*

Thống kê thực tế: **989/2.118 chunk** có đại từ cần xử lý (47%), trong đó **431 chunk** bị sửa văn bản và **148 mention được giữ nguyên** vì mơ hồ, **0 batch lỗi**.

- **Ví dụ từ dữ liệu:** chunk `712ba9de8385e8aa0ac5::c0000`
  > *"3 Safe Haven Information Technology Stocks. **This** makes **it** one of the best safe haven information technology stocks for your portfolio. The overwhelming bulk of Google's income comes…"*

  `unresolved_mentions = ['This (makes it)', 'it', 'this (at end)']`

- **Hiện tượng:** Tiêu đề là một bài listicle nói về *ba* cổ phiếu, còn thân bài lại nhắc tới Google. Đại từ `it` không có tiền ngữ xác định trong cùng chunk — nó có thể trỏ tới Google, tới một trong ba cổ phiếu, hoặc tới cả nhóm ngành. Prompt conservative đã **từ chối phân giải** và đẩy cả ba mention vào `unresolved_mentions`.

- **Hậu quả nếu phân giải sai:** LLM sẽ viết lại thành *"Google makes Google one of the best safe haven stocks"*, và bước NER/RE kế tiếp sinh ra cạnh sai gán cho Google. Nguy hiểm ở chỗ cạnh sai đó **vẫn có đủ provenance** (`source_chunk_id`, `published_date`, `evidence`) nên sanity check `invalid_provenance_edges = 0` **không bắt được**. Đây là lỗi **im lặng**: đồ thị trông sạch nhưng nội dung sai, và sai lệch chỉ lộ ra ở tầng câu trả lời khi đã quá muộn để truy vết.

- **Ví dụ thứ hai** — chunk `916a3ebf144f682693af::c0000`, mention `these systems` trong bài về ransomware tấn công Prospect Medical Holdings: không rõ "these systems" là hệ thống của bệnh viện nào trong "at least three states".

**Cơ chế phòng vệ đã áp dụng:** (1) chỉ phân giải khi tiền ngữ nằm trong cùng chunk, (2) chunk overlap 40 từ để tiền ngữ ít bị cắt ngang, (3) mọi relation bắt buộc có `evidence` để audit ngược, (4) lọc trước bằng regex đại từ — chunk không có đại từ thì không gọi LLM (tiết kiệm 53% chi phí coref mà không mất gì).

---

### 2. Entity Resolution Threshold & Lexical Guard

> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*

- **Ngưỡng cosine similarity:** `ER_THRESHOLD = 0.90`, lấy `top_k = 5` láng giềng (FAISS `IndexFlatIP` trên vector đã normalize).
- **Kết quả:** **907 mention → 898 canonical entity**; bảng audit 15 dòng — 7 `MERGE_VECTOR`, 3 `MERGE_MANUAL`, **5 `REJECT_GUARD`**.

**Lý do chọn 0.90 chứ không phải 0.80:** MiniLM cho điểm rất cao với *mọi* cặp tên công ty công nghệ vì chúng cùng miền ngữ nghĩa. Ở mức 0.80, những cặp hoàn toàn khác nhau cũng lọt vào candidate. Quan trọng hơn: embedding chỉ dùng để **sinh candidate**, quyết định merge cuối cùng luôn thuộc về lexical guard — nên không cần ép threshold quá chặt để bù cho việc thiếu guard.

**Các cặp similarity cao bị Guard chặn:**

| Thực thể A | Thực thể B | Similarity | Lý do chặn | Đánh giá |
|---|---|---|---|---|
| `2013` | `2014` | 0.906 | `version_or_digit_mismatch` | ✅ Chặn đúng |
| `Cloud computing` | `cloud computing service` | 0.903 | `extra_qualifier_tokens:service` | ✅ Chặn đúng |
| `Generative AI Solution` | `generative AI` | 0.909 | `extra_qualifier_tokens:solution` | ✅ Chặn đúng |
| `Polymer Technology and Services LLC` | `Polymer Technology & Services` | 0.927 | `extra_qualifier_tokens:and` | ⚠️ **False reject** |
| `Amazon Web Services (AWS)` | `Amazon Web Services` | 0.929 | `extra_qualifier_tokens:aws` | ⚠️ **False reject** |

**Phân tích cặp `Cloud computing` vs `cloud computing service` (sim 0.903):** hai chuỗi gần như trùng nhau về mặt embedding, nhưng một bên là **công nghệ** còn một bên là **dịch vụ thương mại hoá công nghệ đó**. Nếu gộp, đồ thị mất khả năng phân biệt "công ty X dùng công nghệ Y" với "công ty X bán dịch vụ Y" — hai quan hệ có ý nghĩa kinh doanh hoàn toàn khác.

**Tự phê bình — guard đang quá chặt ở 2 ca:** `Amazon Web Services (AWS)` vs `Amazon Web Services` thực chất **là cùng một thực thể**; guard chặn vì token thừa `aws` không nằm trong danh sách token tổ chức chung. Tương tự với `and` vs `&`. Đây là **false reject** làm phân mảnh đồ thị (giảm recall), tuy nhiên hậu quả nhẹ hơn false merge rất nhiều: false reject chỉ làm mất cạnh, còn false merge **tạo ra sự thật sai và không sửa ngược được** sau khi đã ghi vào graph. Hướng khắc phục: chuẩn hoá `&` → `and`, và bóc acronym trong ngoặc đơn trước khi so token.

**Bốn nhóm nguy hiểm mà guard được thiết kế để chặn (Challenge B):** ticker (`MSFT` vs `Microsoft` — chỉ merge qua `MANUAL_ALIASES`), product chứa tên công ty (`Apple Watch` vs `Apple`), khác phiên bản (`GPT-4` vs `GPT-3`), người trùng họ (`Sam Altman` vs `Steve Altman`). Bảng audit đầy đủ: `outputs/entity_resolution_audit.csv`, cột `reason` ghi rõ căn cứ từng quyết định.

---

### 3. Đồ thị & Super-node Mitigation

> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*

- **Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Microsoft | Company | 15 |
| 2 | Google | Company | 12 |
| 3 | L&T Technology Services | Company | 11 |

**Quan sát quan trọng — bài test suýt pass rỗng:** degree cao nhất chỉ là **15**, thấp hơn rất nhiều so với ngưỡng production `SUPER_NODE_DEGREE = 100`. Nghĩa là trong lần benchmark này **nhánh cắt tỉa không hề chạy**, và nếu chỉ gọi `test_supernode_policy()` thì nó sẽ PASS mà **không chứng minh được gì cả**.

Vì vậy cell **5.1c** ép nhánh này chạy bằng cách hạ ngưỡng xuống dưới degree thật:

| | |
|---|---|
| Node test | Microsoft (degree thật = 15) |
| Ngưỡng hạ | `degree > 14` → cap 5 cạnh |
| Không áp trần | 15 cạnh |
| Có áp trần | **5 cạnh** |
| Cắt đúng cap | ✅ PASS |
| Giữ đúng cạnh mới nhất | ✅ PASS |

Ngày của 5 cạnh được giữ: `2023-09-21, 2023-09-14, 2023-08-09, 2023-06-18, 2023-06-18` — đúng là 5 ngày mới nhất, 10 cạnh cũ hơn (tới `2022-12-14`) bị loại. Kết quả lưu tại `outputs/supernode_policy_test.csv`.

- **Ưu điểm & Rủi ro của Temporal Mitigation:**
  - *Ưu điểm:* Chặn bùng nổ context — không có cap, một node degree 10.000 sẽ kéo toàn bộ láng giềng vào prompt, vượt context window và làm nhiễu chìm mất tín hiệu. Ưu tiên theo thời gian cũng đúng với đặc thù tin công nghệ: trạng thái mới nhất thường là trạng thái đúng (*"ai đang lãnh đạo X"*, *"X vừa mua lại ai"*).
  - *Rủi ro:* Cắt theo `published_date DESC` **xoá mất dấu vết lịch sử**, tức là làm hỏng đúng nhóm câu hỏi cross-doc mà GraphRAG lẽ ra thắng. Rủi ro thứ hai: bài không có ngày xuất bản mang `published_date = ''` nên bị đẩy xuống cuối và gần như **không bao giờ được chọn** — một dạng thiên lệch âm thầm.
  - *Cải tiến đề xuất:* thay "top-50 mới nhất" bằng **quota lai** — 25 cạnh mới nhất + 15 cạnh confidence cao nhất + 10 cạnh cũ nhất; hoặc xếp hạng cạnh theo độ liên quan ngữ nghĩa giữa `evidence` và câu hỏi thay vì thuần thời gian.

Ngoài degree cap còn hai lớp chặn nữa: `GLOBAL_EDGE_CAP = 250` cạnh mỗi truy vấn, `MAX_GRAPH_CONTEXT_CHARS = 14.000` ký tự, và `MAX_EXPANDED_NODES = 60` node mở rộng.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Golden Dataset của giảng viên: **25 câu** — 2 `factoid`, 12 `multi-hop`, 11 `cross-doc`, gold answer neo vào đúng 5.000 dòng đầu.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 3.72 | **3.84** | +0.12 | GraphRAG nhỉnh hơn, nhưng chênh lệch nhỏ hơn phương sai giữa các lần chạy |
| **Faithfulness (1–5)** | 4.12 | **4.20** | +0.08 | Subgraph có provenance giúp model bớt suy diễn |
| **Multi-hop Reasoning (1–5)** | 3.72 | **3.84** | +0.12 | Lợi thế tập trung ở cross-doc chứ không đều |
| **Latency trung bình (s)** | 2.511 | **2.508** | −0.004 | Ngang nhau ở khâu sinh; GraphRAG còn tốn thêm 1 lần LLM trích seed + N truy vấn Cypher |
| **Token usage trung bình** | **769** | 1.377 | +608 | GraphRAG đắt hơn **~1,8 lần** — đây mới là cái giá thật |

#### Tách theo nhóm câu hỏi (điều bảng tổng che mất):

| Nhóm | Comprehensiveness (Flat → Graph) | Faithfulness (Flat → Graph) | Kết luận |
|---|---|---|---|
| `cross-doc` | 3.727 → **4.091** | 4.364 → **4.545** | **GraphRAG thắng rõ** |
| `factoid` | 5.000 → 5.000 | 5.000 → 5.000 | **Hoà tuyệt đối** — graph chỉ tốn thêm token |
| `multi-hop` | 3.500 → 3.417 | 3.750 → 3.750 | **Flat nhỉnh hơn chút** |

**Cảnh báo về độ tin cậy thống kê:** pipeline được chạy hai lần với cùng cấu hình (`temperature = 0`), delta Faithfulness lần lượt là **+0.40** rồi **+0.08**. Với n = 25 và LLM-as-a-Judge, **phương sai giữa các lần chạy ngang ngửa chính hiệu ứng đang đo**. Kết luận đáng tin duy nhất là *xu hướng theo nhóm* — cross-doc thắng, factoid hoà, token gấp ~1,8 lần — chứ không phải con số delta tuyệt đối. Muốn khẳng định mạnh hơn cần mở rộng golden set và chạy nhiều seed rồi báo cáo khoảng tin cậy.

**Một hạn chế phương pháp cần nêu:** generator và judge hiện là **cùng một model** (`gpt-4o-mini`) nên có nguy cơ thiên vị tự chấm. So sánh Flat vs Graph vẫn công bằng (chung generator, chung judge), nhưng *điểm tuyệt đối* nên đọc dè dặt. Khắc phục: dùng judge mạnh hơn và khác họ model.

#### Phân tích 2 Ca lỗi Điển hình:

**1. Ca lỗi Flat RAG thất bại (GraphRAG thành công) — `G5000-50`, nhóm `multi-hop`, Δ = +2.33**

- *Câu hỏi:* "Compare the chip-related AI positioning of NVIDIA, AMD, and Intel in the selected 5,000-row scope. What distinct signal is reported for each?"
- *Tín hiệu retrieval:* GraphRAG match được 2 seed, dùng **12 cạnh**.
- *Tại sao Flat RAG thất bại?* Câu hỏi đòi thông tin về **ba công ty nằm ở ba bài báo khác nhau**. Top-k = 6 chunk của vector search bị "hút" về cụm Intel/AMD (chip-stacking, 3D V-Cache) và **bỏ sót hoàn toàn NVIDIA** — đúng biểu hiện phân mảnh ngữ cảnh: mỗi chunk riêng lẻ đều liên quan, nhưng tập top-k không phủ hết các thực thể được hỏi. Judge nhận xét: *"fails to accurately represent NVIDIA's position, which was explicitly mentioned in the reference"*.
- *GraphRAG giải quyết thế nào?* BFS xuất phát từ các seed entity chip, thu 12 cạnh trải trên cả ba công ty **bất kể chúng nằm ở bài báo nào**. Cấu trúc đồ thị đảm bảo độ phủ theo *thực thể*, trong khi vector search chỉ đảm bảo độ phủ theo *độ tương đồng văn bản*. Judge: *"closely aligning with the reference answer, accurately captures NVIDIA's leadership"*.

**2. Ca lỗi GraphRAG thất bại — `G5000-43`, nhóm `cross-doc`, Δ = −2.00**

- *Câu hỏi:* "Which came first in the selected HPE timeline: the Axis Security acquisition agreement or the LLM-focused cloud service announcement, and what does that ordering show?"
- *Tín hiệu retrieval:* match được 2 seed nhưng **chỉ thu được 2 cạnh**.
- *Nguyên nhân:* Bước NER/RE **không trích được cạnh nào cho sự kiện "HPE ra mắt dịch vụ cloud cho LLM"** — đó là một thông báo sản phẩm, không khớp gọn với bất kỳ quan hệ nào trong allowlist 8 loại (`ACQUIRED, DEVELOPED, INVESTED_IN, FOUNDED, WORKED_AT, PARTNERED_WITH, USES, LEADS`). Hệ quả: subgraph chỉ có mốc Axis Security. GraphRAG trả lời trung thực rằng *"the LLM-focused cloud service announcement is not provided in the context"* — **không bịa, nhưng không trả lời được**. Trong khi đó Flat RAG lấy trúng chunk văn bản thô chứa cả hai mốc và suy ra được thứ tự.
- *Bài học:* đây là **cái giá của việc nén văn bản thành triple**. Allowlist quá hẹp làm mất thông tin sự kiện, và graph không thể suy luận về cái nó không có cạnh. Đáng chú ý: GraphRAG "thất bại" ở đây theo cách **an toàn** (thừa nhận thiếu bằng chứng) chứ không hallucinate.
- *Đề xuất khắc phục:* (1) bổ sung quan hệ `ANNOUNCED` / `LAUNCHED` vào allowlist để bắt sự kiện ra mắt sản phẩm; (2) hybrid context đã có sẵn phần `=== VECTOR ===` — nên tăng trọng số phần này khi subgraph thu được quá ít cạnh (ví dụ < 5), thay vì giữ k = 4 cố định; (3) cơ chế self-correction (bonus C) chính là hướng đi đúng cho ca này.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*

- **Đánh đổi Quality vs Cost vs Latency:**
  GraphRAG tốn **~1,8 lần token** (1.377 vs 769) để đổi lấy **+0.12 điểm chất lượng trung bình** — một cái giá đắt nếu áp dụng đại trà. Nhưng con số trung bình che mất bức tranh thật: ở nhóm `factoid` GraphRAG **hoà tuyệt đối 5.0/5.0** trong khi vẫn đốt gấp đôi token (lãng phí thuần tuý), còn ở `cross-doc` nó thắng rõ. Chi phí thật của GraphRAG còn lớn hơn con số latency 2.508s ở trên, vì bảng chỉ đo lần gọi LLM sinh câu trả lời; chưa tính 1 lần LLM trích seed entity và các truy vấn Cypher cho BFS.
  Ngoài ra còn **indexing overhead** một lần: 133 request trích xuất KG + 124 request coref để dựng đồ thị — Flat RAG chỉ cần tính embedding.
  **Kết luận vận hành:** dùng **router** — câu factoid đi thẳng Flat RAG, chỉ multi-hop/cross-doc mới kích hoạt GraphRAG. Đây là kiến trúc rẻ nhất mà vẫn giữ được phần thắng.

- **Quyết định từ chối AI Coding Agent:**
  1. **Pairwise cosine $O(N^2)$ cho near-dedup** — với 2.118 bài thì chạy được, nhưng ở 350MB (~hàng trăm nghìn bài) là ~$10^{10}$ phép so sánh. Đã thay bằng **MinHash 128 perm + LSH 32 band**, chỉ tính Jaccard thật trên **504 cặp candidate** (phát hiện 12 bài trùng lặp gần).
  2. **`MERGE` từng dòng trong vòng lặp Python** — mỗi row một round-trip tới AuraDB. Đã thay bằng `UNWIND $rows AS row` batch 1000: nạp 898 node + 554 cạnh chỉ mất **1,8 giây**.
  3. **Hạ threshold entity resolution xuống 0.75 để "gộp được nhiều hơn"** — tăng recall nhưng tạo false merge im lặng, loại lỗi **không sửa ngược được** sau khi đã ghi vào graph.
  4. **Bỏ lexical guard cho gọn** — guard chính là thứ duy nhất chặn `Apple Watch` gộp vào `Apple`.
  5. **Dùng `apoc.*` cho traversal** — không đảm bảo AuraDB free tier có plugin; giữ Cypher thuần để notebook chạy được ở mọi môi trường.
  6. **Chấp nhận `test_supernode_policy()` PASS là đủ** — đây là cái bẫy nguy hiểm nhất: bài test pass *rỗng* vì không node nào chạm ngưỡng. Đã bổ sung cell 5.1c hạ ngưỡng để thực sự chứng minh chính sách hoạt động.

  **Nguyên tắc kiểm soát:** Agent viết code, nhưng **threshold, allowlist schema và chính sách cắt tỉa đều phải có bảng audit và test kiểm chứng** — `entity_resolution_audit.csv`, `near_dedup_audit.csv`, `supernode_policy_test.csv`, `test_edge_provenance()`.

- **Giải pháp scale 350MB (~100.000 bài):**
  Bottleneck đầu tiên **không phải Neo4j** mà là **bước gọi LLM trích xuất NER+RE**. Bằng chứng trực tiếp từ chính lab này: hạn mức token của nhà cung cấp là thứ chặn pipeline trước tiên, không phải CPU hay database.
  1. **LLM extraction:** lọc trước bằng NER rẻ tiền (spaCy) để chỉ gửi chunk *có thực thể*; gộp batch lớn hơn để chia đều overhead prompt; chạy song song nhiều worker với hàng đợi có retry/backoff; **cache theo hash chunk** để không trả tiền hai lần (lab này đã dùng, và chính nó cứu được ~190k token khi phải đổi nhà cung cấp giữa chừng).
  2. **Embedding + FAISS in-memory:** `IndexFlatIP` là brute-force và giữ toàn bộ vector trong RAM → chuyển sang `IndexIVFFlat`/HNSW hoặc vector store ngoài (Qdrant/pgvector).
  3. **Coreference toàn corpus:** thay LLM bằng model coref chuyên dụng (fastcoref); lab này đã giảm 53% chi phí chỉ bằng bộ lọc regex đại từ.
  4. **Neo4j write:** batch 1000 vẫn ổn; lớn hơn thì dùng `apoc.periodic.iterate` hoặc `neo4j-admin import` offline cho lần nạp đầu.
  5. **Super-node:** ở quy mô thật Google/Microsoft sẽ có degree hàng chục nghìn → tỉa theo cửa sổ thời gian và dùng community-level summary thay vì đọc cạnh thô.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `needs_coref()` | 989/2.118 chunk có đại từ; 431 chunk được sửa, **148 mention cố tình giữ nguyên**. Giữ nguyên khi mơ hồ quan trọng hơn phân giải bằng mọi giá |
| **Near-dedup (Challenge A)** | Module 1 | `minhash_signature()`, `lsh_candidate_pairs()` | 504 cặp candidate → 12 cặp trùng thật. LSH chỉ lo *recall* của candidate, Jaccard thật mới quyết định *precision* — hai ngưỡng khác nhau, dễ nhầm |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | 555 triple, 100% có `evidence`, confidence TB 0.977. Nhưng allowlist 8 quan hệ **quá hẹp** — chính nó gây ra ca lỗi G5000-43 |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | 898 node + 554 cạnh trong **1,8s**; 0 cạnh thiếu provenance |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `merge_guard()`, `UF` | 907 → 898 entity. Guard chặn 5 cặp, trong đó **2 cặp là false reject** — guard đang hơi chặt |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `degree_and_edges()` | Degree max chỉ 15 → nhánh cắt tỉa không tự chạy, phải hạ ngưỡng để chứng minh (cell 5.1c) |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `judge_answer_safe()` | 25 câu × 2 hệ thống. Phát hiện quan trọng: **phương sai giữa 2 lần chạy ≈ hiệu ứng đo được** |

### Bonus đã hoàn thành

| Bonus | Kết quả định lượng |
|---|---|
| **Near-Dedup** (MinHash/LSH) | 504 cặp candidate, 12 cặp trùng gần, không dùng $O(N^2)$ |
| **Community Detection + Global Search** | **364 community** (NetworkX greedy modularity), 5 community report do LLM tóm tắt, truy vấn vĩ mô chạy trên report |
| **Self-Correction Retrieval** | 3 câu test: 2 câu phải leo lên `hop3+vector`, 1 câu đủ ở `hop2`. Chất lượng **1.33 → 2.55** |

---

### 2. Quá trình Debugging & Bài học

- **Lỗi phức tạp nhất — hết hạn mức token giữa chừng:** Groq free tier giới hạn **200.000 token/ngày** (không chỉ 8.000 token/phút như tôi tưởng ban đầu). Bước coreference đã tiêu ~193k, khiến bước trích xuất — phần quan trọng nhất — không chạy nổi một request nào.
  *Cách xử lý:* (1) đo lại và xác định nút thắt thật là **TPD chứ không phải TPM**; (2) thử `gpt-oss-20b` và `qwen3.6-27b` (hạn mức riêng) nhưng cả hai **không sinh nổi JSON đúng schema**; (3) chuyển phần còn lại sang `gpt-4o-mini`, đồng thời **tra cứu cache bằng key cũ trước** để không phải trả tiền lại cho ~190k token coref đã hoàn thành.
  *Bài học:* cache theo hash nội dung không chỉ tiết kiệm tiền mà còn là **lớp bảo hiểm khi phải đổi nhà cung cấp giữa chừng**. Và luôn đọc kỹ *cả hai* trục rate limit trước khi lên kế hoạch chạy.

- **Bẫy đo lường — latency giả bằng 0:** sau khi bật cache, lần chạy lại cho `Latency = 0.00s` cho cả hai hệ thống, làm nguyên một cột trong bảng benchmark trở nên vô nghĩa. *Cách xử lý:* bỏ cache riêng cho bước sinh câu trả lời (50 lần gọi) để đo độ trễ API thật, giữ cache cho các bước đắt tiền khác. *Bài học:* tối ưu chi phí có thể âm thầm phá hỏng phép đo.

- **Bẫy đánh giá — bài test pass rỗng:** `test_supernode_policy()` PASS nhưng không chứng minh gì vì không node nào chạm ngưỡng 100. *Bài học:* một assertion xanh chưa chắc là bằng chứng; phải kiểm tra xem nhánh code cần test **có thực sự được chạy** hay không.

- **Schema dataset không như dự đoán:** dump thực tế có cột `description` rất ngắn (trung vị 21 từ, 2.306/5.000 dòng gần như rỗng) trong khi thực thể chủ yếu nằm ở `title`. *Cách xử lý:* ghép `title` vào `text` trước khi chunk — nếu không, cả Flat RAG lẫn NER/RE đều mất phần lớn thực thể.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

> ✍️ *Phần dưới là bản nháp dựa trên kiến trúc đã dựng trong lab — cần thay bằng bài toán thực tế của bạn trước khi nộp.*

- **Tên đồ án / Dự án:** `[Điền tên dự án của bạn]`
- **Đặc thù bài toán & Lý do chọn giải pháp:** GraphRAG chỉ xứng đáng khi câu hỏi thực tế đòi hỏi **nối nhiều quan hệ qua nhiều tài liệu** (truy vết chuỗi sở hữu/đầu tư, phân tích ảnh hưởng giữa các thực thể, so sánh dòng thời gian). Nếu ~90% câu hỏi chỉ là tra cứu một đoạn văn bản thì Flat RAG rẻ hơn và ít điểm hỏng hơn — số liệu nhóm `factoid` (hoà 5.0/5.0 nhưng tốn gấp đôi token) chứng minh điều đó bằng thực nghiệm.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Technology`, `Product`, `Event` (đều mang base label `Entity`)
  - Relations: giữ allowlist nhỏ nhưng **phải phủ được sự kiện** — rút kinh nghiệm từ ca lỗi G5000-43, cần thêm `ANNOUNCED` / `LAUNCHED` bên cạnh 8 quan hệ hiện có. Mọi cạnh bắt buộc có `source_chunk_id`, `published_date`, `evidence`, `confidence`
- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Super-node: quota lai (mới nhất + confidence cao + cũ nhất) thay vì thuần `published_date DESC`; community summary thay cho cạnh thô khi degree quá lớn
  - Entity Resolution: giữ vector ≥ 0.90 chỉ để sinh candidate, quyết định bằng lexical guard; **duyệt tay top-N cặp similarity cao** trước khi cho merge tự động; chuẩn hoá `&`/acronym trong ngoặc để giảm false reject
  - Đo lường: Golden Dataset chia 3 nhóm, chạy lại mỗi lần đổi prompt/threshold; theo dõi cả quality lẫn latency/token; **chạy nhiều lần và báo cáo khoảng dao động**, không tin vào một con số delta duy nhất

---

## 🎯 TỰ ĐÁNH GIÁ

> ✍️ *Điểm tự chấm nên do bạn tự điền — dưới đây chỉ là gợi ý kèm bằng chứng để bạn cân nhắc.*

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | `[ ]` | Đã dựng đủ 5 module, giải thích được vì sao cross-doc thắng còn factoid hoà |
| Khả năng kiểm soát AI Coding Agent | `[ ]` | Từ chối 6 đề xuất có hại, mỗi quyết định đều có bảng audit hoặc test kiểm chứng |
| Chất lượng đồ thị tri thức xây dựng | `[ ]` | 898 node / 554 cạnh, 0 cạnh thiếu provenance, 100% có evidence — nhưng allowlist còn hẹp và guard còn 2 false reject |
| Khả năng phân tích và debug hệ thống | `[ ]` | Xử lý được 3 bẫy: hết quota giữa chừng, latency giả bằng 0, test pass rỗng |

---

## 📎 Phụ lục — File kết quả

| Đường dẫn | Nội dung |
|---|---|
| `outputs/graphrag_eval_results.csv` | Kết quả chi tiết từng câu trong 25 câu golden |
| `outputs/graphrag_vs_flatrag_summary.csv` | Bảng so sánh tách theo nhóm câu hỏi |
| `outputs/graphrag_vs_flatrag_overall.csv` | Bảng tổng hợp toàn bộ golden set |
| `outputs/entity_resolution_audit.csv` | 15 quyết định merge/reject kèm lý do |
| `outputs/near_dedup_audit.csv` | 504 cặp candidate từ MinHash/LSH |
| `outputs/supernode_policy_test.csv` | Chứng minh chính sách super-node bằng ngưỡng hạ thấp |
| `outputs/self_correction_before_after.csv` | Định lượng trước/sau self-correction |
| `outputs/community_reports.csv` | 5 community report cho global search |
| `outputs/submission_checklist.csv` | 20/20 mục PASS |
| `reports/technical_defense.md` | Bản thuyết minh 10 câu (chi tiết hơn Phần 1) |
| `reports/failure_analysis.md` | Phân tích ca lỗi theo root-cause |
| `Day19_..._EXECUTED.ipynb` | Notebook đã chạy đầy đủ output |
