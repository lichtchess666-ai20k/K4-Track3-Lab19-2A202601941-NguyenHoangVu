# Failure Analysis — Lab 19: GraphRAG vs Flat RAG

**Sinh viên:** Nguyen Hoang Vu (2A202601941)
**Cấu hình:** embedding = `sentence-transformers/all-MiniLM-L6-v2`, generator = `gpt-4o-mini`, judge = `openai/gpt-4o-mini`
**Golden set:** 25 câu — {'multi-hop': 12, 'cross-doc': 11, 'factoid': 2}

Cả hai kiến trúc dùng **chung embedding và chung generator**, chỉ khác cách lấy ngữ cảnh. Vì vậy mọi chênh lệch điểm dưới đây đều quy được về tầng retrieval.

### Ca 1 — GraphRAG thắng Flat RAG

- **Câu hỏi (G5000-50 / multi-hop):** Compare the chip-related AI positioning of NVIDIA, AMD, and Intel in the selected 5,000-row scope. What distinct signal is reported for each?
- **Reference:** NVIDIA reported that use of its chips for AI services such as chatbots had gone 'through the roof' and its sales outlook beat expectations. AMD was named as a possible supplier of new AI chips to AWS, but AWS had made no final decision. Intel said its 3D V-Cache/chip-stacking approach would differ from AMD's and would be used for Intel processors and offered to foundry customers.
- **Điểm Flat:** 2.00 · **Điểm Graph:** 4.33 · **Delta:** +2.33
- **Tín hiệu retrieval:** seed khớp = 2, cạnh dùng = 12, super-node event = 0, độ dài context flat/graph = 2019/3947 ký tự

**Câu trả lời Flat RAG**

> In the context of AI chip positioning:

- **Intel** is focusing on innovative chip-stacking technology, specifically its 3D V-Cache, which it plans to utilize for its processors and offer to third-party customers through its foundry business [chunk_id=73f6d5535f26475909a8::c0000]. This indicates a strategy aimed at enhancing performance and flexibility in chip design.

- **AMD** is being considered by Amazon Web Services for its new AI chips, suggesting that AMD is gaining traction in the AI market, particularly in cloud computing applications [chunk_id=ec3ce18d568e33867724::c0000]. This reflects a growing recognition of AMD's capabilities in AI hardware.

- **NVIDIA** is not explicitly mentioned in the provided context, so no distinct signal can be reported for its AI positioning based on the available information.

Overall, Intel is innovating in chip technology, AMD is being recognize

**Câu trả lời GraphRAG**

> In the context of chip-related AI positioning:

1. **NVIDIA**: The company is heavily focused on AI, with its chips being increasingly utilized for AI services, such as chatbots. The CEO noted a significant surge in the use of NVIDIA chips for AI applications, indicating strong market demand and leadership in this area [chunk_id=64f859127d105927b7dc::c0000].

2. **AMD**: AMD is in discussions with Amazon Web Services regarding its new AI chips, suggesting that it is positioning itself as a viable option for cloud-based AI solutions. However, a final decision from AWS has not yet been made, indicating that AMD is still in the evaluation phase for its AI offerings [chunk_id=ec3ce18d568e33867724::c0000].

3. **Intel**: Intel is taking a different approach with its 3D V-Cache technology, which it plans to use for its processors and offer to third-party customers. This indicates a focus on en

**Judge rationale (flat):** The candidate provides some relevant information about Intel and AMD but fails to accurately represent NVIDIA's position, which was explicitly mentioned in the reference. The assessment of NVIDIA's AI positioning is incomplete, leading to a lack of comprehensiveness. Additionally, the candidate does not fully capture the distinct signals for each company as outlined in the reference, particularly for NVIDIA.

**Judge rationale (graph):** The candidate provides a detailed comparison of NVIDIA, AMD, and Intel's AI chip positioning, closely aligning with the reference answer. It accurately captures NVIDIA's leadership in AI chip usage, AMD's potential role with AWS, and Intel's unique 3D V-Cache approach. However, it could enhance comprehensiveness by explicitly stating the distinct signals reported for each company, which would improve clarity and completeness.

**Root-cause analysis**

1. *Triệu chứng:* chênh lệch điểm +2.33 giữa hai kiến trúc trong khi generator và embedding **hoàn toàn giống nhau**, nên nguyên nhân nằm ở tầng **retrieval**, không phải ở model sinh.
2. *Bước hỏng:* seed matching hoạt động; điểm nghi vấn tiếp theo là số cạnh thu được (12) và ngưỡng cắt super-node.
3. *Bằng chứng:* đối chiếu `graph_debug['diagnostics']` của câu hỏi này và các trích dẫn `[chunk_id=...]` trong câu trả lời để xem model dựa vào cạnh nào.
4. *Hướng sửa:* giữ nguyên cấu hình: graph traversal đang bù được phần ngữ cảnh bị phân mảnh của vector search.


### Ca 2 — Flat RAG thắng / GraphRAG gặp khó

- **Câu hỏi (G5000-43 / cross-doc):** Which came first in the selected HPE timeline: the Axis Security acquisition agreement or the LLM-focused cloud service announcement, and what does that ordering show?
- **Reference:** The Axis Security acquisition agreement came first on March 2, 2023; the HPE LLM-focused cloud service story followed on June 21, 2023. The ordering shows HPE expanding security capabilities before the later AI-cloud service initiative in this dataset slice.
- **Điểm Flat:** 4.33 · **Điểm Graph:** 2.33 · **Delta:** -2.00
- **Tín hiệu retrieval:** seed khớp = 2, cạnh dùng = 2, super-node event = 0, độ dài context flat/graph = 2366/2229 ký tự

**Câu trả lời Flat RAG**

> The Axis Security acquisition agreement came first on March 2, 2023, before the LLM-focused cloud service announcement, which is not explicitly mentioned in the provided context but can be inferred to have occurred later. This ordering shows that HPE prioritized enhancing its security capabilities through acquisition before expanding its cloud service offerings, indicating a strategic focus on strengthening security as a foundational element of its service portfolio.

**Câu trả lời GraphRAG**

> The Axis Security acquisition agreement came first on March 2, 2023, while the LLM-focused cloud service announcement is not provided in the context. Therefore, the ordering shows that HPE prioritized enhancing its security capabilities through the acquisition before potentially expanding into LLM-focused services. However, without specific details on the LLM-focused cloud service announcement, further conclusions cannot be drawn.

**Judge rationale (flat):** The candidate accurately states that the Axis Security acquisition agreement came first on March 2, 2023, and implies that the LLM-focused cloud service announcement occurred later. However, it does not explicitly mention the date of the LLM-focused cloud service announcement, which is June 21, 2023, as provided in the reference. The reasoning about HPE's strategic focus on security before cloud services is well articulated, but the lack of specific date reference slightly affects comprehensiveness.

**Judge rationale (graph):** The candidate correctly identifies the Axis Security acquisition date but fails to provide the date for the LLM-focused cloud service announcement, which is crucial for a complete answer. While it mentions the prioritization of security capabilities, it does not fully explain the implications of the ordering as stated in the reference. The reasoning lacks depth and clarity, impacting the overall accuracy.

**Root-cause analysis**

1. *Triệu chứng:* chênh lệch điểm -2.00 giữa hai kiến trúc trong khi generator và embedding **hoàn toàn giống nhau**, nên nguyên nhân nằm ở tầng **retrieval**, không phải ở model sinh.
2. *Bước hỏng:* seed matching hoạt động; điểm nghi vấn tiếp theo là số cạnh thu được (2) và ngưỡng cắt super-node.
3. *Bằng chứng:* đối chiếu `graph_debug['diagnostics']` của câu hỏi này và các trích dẫn `[chunk_id=...]` trong câu trả lời để xem model dựa vào cạnh nào.
4. *Hướng sửa:* tăng max_hops/edge_limit hoặc bổ sung quan hệ còn thiếu vào ALLOWED_RELATIONS — subgraph hiện tại chưa chứa đủ mắt xích.


## Các failure mode hệ thống đã chặn được

| Failure mode | Cơ chế chặn | Bằng chứng trong notebook |
|---|---|---|
| False coreference → false edge | Prompt conservative, chỉ resolve khi tiền ngữ nằm trong cùng chunk; phần mơ hồ đẩy vào `unresolved_mentions` | Cell 1.7 — 148 mention giữ nguyên, không đoán |
| Near-duplicate làm lệch tần suất entity | MinHash 128 perm + LSH 32 band, giữ lại cặp có Jaccard ≥ 0.8 | Cell 1.5b — loại 0 bài trùng lặp gần |
| False merge entity | Vector ≥ 0.9 chỉ để sinh candidate; quyết định cuối do lexical guard (ticker / hậu tố / product / trùng họ) | Cell 2.2 — audit {'MERGE_VECTOR': 7, 'REJECT_GUARD': 5, 'MERGE_MANUAL': 3} |
| Edge thiếu provenance | Bắt buộc `source_chunk_id` + `published_date` ngay trong câu `UNWIND` | Cell 2.4 và 5.1 — `invalid_provenance_edges = 0` |
| Super-node làm nổ context | degree > 100 → tối đa 50 cạnh mới nhất; trần toàn cục 250 cạnh và 14000 ký tự | Cell 5.1 + 5.1c — node đậm đặc nhất `Microsoft` có degree 15; hạ ngưỡng xuống 14 thì 15 cạnh bị cắt còn 5, đúng các cạnh mới nhất |

## Ghi chú về ngưỡng super-node

Trên subset của giờ lab, node có degree cao nhất là **Microsoft (15)**, thấp hơn nhiều so với ngưỡng production `SUPER_NODE_DEGREE = 100`. Nghĩa là nhánh cắt tỉa **không hề chạy** trong lần benchmark này, và một bài kiểm tra chỉ gọi `test_supernode_policy()` sẽ pass một cách rỗng — không chứng minh được gì.

Vì vậy cell 5.1c hạ ngưỡng xuống `degree > 14` với cap 5 để ép nhánh này chạy: 15 cạnh bị cắt còn 5, và tập giữ lại đúng bằng 5 cạnh có `published_date` mới nhất (cắt đúng cap: True, giữ đúng cạnh mới nhất: True).

Ở quy mô thật (350MB), Google/Microsoft sẽ có degree hàng chục nghìn và nhánh này chạy liên tục; lúc đó rủi ro đã phân tích ở mục 5 của `technical_defense.md` — mất dấu vết lịch sử — mới trở thành vấn đề thực sự.
