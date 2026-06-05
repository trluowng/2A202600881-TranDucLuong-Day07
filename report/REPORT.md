# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Đức Lương
**Nhóm:** F4
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai đoạn text có cosine similarity cao khi vector embedding của chúng trỏ gần cùng một hướng, nghĩa là mô hình biểu diễn chúng là gần nhau về chủ đề hoặc ý nghĩa. Trong retrieval, điểm cao thường cho thấy chunk đó có khả năng liên quan đến query.

**Ví dụ HIGH similarity:**
- Sentence A: Python is a high-level programming language.
- Sentence B: Python is used to write software and scripts.
- Tại sao tương đồng: Cả hai câu đều nói về Python và vai trò của nó trong lập trình.

**Ví dụ LOW similarity:**
- Sentence A: Vector databases store embeddings for similarity search.
- Sentence B: The recipe needs fresh basil and tomatoes.
- Tại sao khác: Một câu nói về hệ thống dữ liệu/embedding, câu còn lại nói về nấu ăn.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector nên phù hợp để so sánh ý nghĩa, ít bị ảnh hưởng bởi độ lớn tuyệt đối của vector. Với text embeddings, hai câu có cùng ý nhưng độ dài/độ mạnh biểu diễn khác nhau vẫn nên được xem là gần nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, nên số chunk tăng từ 23 lên 25. Overlap nhiều hơn giúp giữ ngữ cảnh giữa hai chunk liền kề, nhưng đổi lại tốn thêm lưu trữ và có thể tạo nhiều kết quả trùng lặp hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Y tế / bệnh truyền nhiễm, tập trung vào các loại sốt thường gặp và phân biệt sốt xuất huyết.

**Tại sao nhóm chọn domain này?**
> Bộ tài liệu mới merge gồm các bài Vinmec tiếng Việt có cùng chủ đề "sốt", đặc biệt là sốt xuất huyết, sốt virus, sốt rét, sốt phát ban và hướng dẫn xử trí sốt ở trẻ em. Domain này phù hợp để kiểm thử retrieval vì các tài liệu có nhiều thuật ngữ giống nhau nhưng khác ý nghĩa lâm sàng, nên chunking và metadata ảnh hưởng rõ đến kết quả tìm kiếm.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Bệnh sốt tinh hồng nhiệt (sốt Scarlet) là bệnh gì? | vinmec.com | 5936 | `specialty=truyen_nhiem`, `disease_type=sot_scarlet`, `type=trieu_chung`, `audience=chung`, `language=vi` |
| 2 | Bị sốt mấy ngày thì xét nghiệm sốt xuất huyết? | vinmec.com | 6473 | `specialty=truyen_nhiem`, `disease_type=sot_xuat_huyet`, `type=huong_dan`, `audience=chung`, `language=vi` |
| 3 | Hướng dẫn phân biệt sốt virus với sốt xuất huyết | vinmec.com | 6416 | `specialty=truyen_nhiem`, `disease_type=sot_virus,sot_xuat_huyet`, `type=phan_biet`, `audience=chung`, `language=vi` |
| 4 | Phân biệt sốt rét và sốt xuất huyết | vinmec.com | 5441 | `specialty=truyen_nhiem`, `disease_type=sot_ret,sot_xuat_huyet`, `type=phan_biet`, `audience=chung`, `language=vi` |
| 5 | Phân biệt sốt thường, sốt virus và sốt xuất huyết | vinmec.com | 5955 | `specialty=truyen_nhiem`, `disease_type=sot_virus,sot_xuat_huyet`, `type=phan_biet`, `audience=chung`, `language=vi` |
| 6 | Sốt nóng lạnh nhức mỏi đau đầu, có phải sốt virus? | vinmec.com | 6340 | `specialty=truyen_nhiem`, `disease_type=sot_virus`, `type=trieu_chung`, `audience=chung`, `language=vi` |
| 7 | Sốt phát ban khác sốt xuất huyết như thế nào? | vinmec.com | 4362 | `specialty=truyen_nhiem`, `disease_type=sot_xuat_huyet`, `type=phan_biet`, `audience=chung`, `language=vi` |
| 8 | Sốt xuất huyết và sốt xuất huyết nặng | vinmec.com | 5723 | `specialty=truyen_nhiem`, `disease_type=sot_xuat_huyet`, `type=trieu_chung`, `audience=chung`, `language=vi` |
| 9 | Trẻ sốt đến đâu mới phải uống thuốc hạ sốt? | vinmec.com | 5695 | `specialty=truyen_nhiem`, `disease_type=sot_chung`, `type=dieu_tri`, `audience=tre_em`, `language=vi` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `disease_type` | string | `sot_virus,sot_xuat_huyet` | Giúp gom/lọc theo bệnh hoặc nhóm bệnh khi nhiều bài cùng nói về "sốt". |
| `type` | string | `phan_biet`, `trieu_chung`, `dieu_tri` | Hữu ích để tách query hỏi phân biệt, triệu chứng, hướng dẫn điều trị/xử trí. |
| `audience` | string | `chung`, `tre_em` | Giúp query liên quan trẻ em chỉ tìm trong tài liệu đúng đối tượng. |
| `source` | string | `vinmec.com` | Truy vết nguồn tài liệu và hỗ trợ grounding khi trình bày kết quả. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Hướng dẫn phân biệt sốt virus với sốt xuất huyết | FixedSizeChunker (`fixed_size`) | 24 | 296.1 | Trung bình: giữ đúng size nhưng có thể cắt ngang câu |
| Hướng dẫn phân biệt sốt virus với sốt xuất huyết | SentenceChunker (`by_sentences`) | 20 | 319.4 | Tốt: giữ trọn câu và checklist triệu chứng |
| Hướng dẫn phân biệt sốt virus với sốt xuất huyết | RecursiveChunker (`recursive`) | 33 | 193.0 | Tốt ở ranh giới đoạn/câu nhưng tạo nhiều chunk nhỏ |
| Sốt xuất huyết và sốt xuất huyết nặng | FixedSizeChunker (`fixed_size`) | 22 | 288.8 | Trung bình |
| Sốt xuất huyết và sốt xuất huyết nặng | SentenceChunker (`by_sentences`) | 19 | 299.3 | Tốt |
| Sốt xuất huyết và sốt xuất huyết nặng | RecursiveChunker (`recursive`) | 28 | 202.6 | Khá, nhưng có thể tách danh sách triệu chứng quá nhỏ |
| Trẻ sốt đến đâu mới phải uống thuốc hạ sốt? | FixedSizeChunker (`fixed_size`) | 21 | 299.8 | Trung bình |
| Trẻ sốt đến đâu mới phải uống thuốc hạ sốt? | SentenceChunker (`by_sentences`) | 23 | 245.9 | Tốt, giữ nguyên câu hướng dẫn 38,5°C |
| Trẻ sốt đến đâu mới phải uống thuốc hạ sốt? | RecursiveChunker (`recursive`) | 29 | 194.7 | Khá, nhưng nhiều chunk nhỏ |

### Strategy Của Tôi

**Loại:** SentenceChunker (`max_sentences_per_chunk=3`)

**Mô tả cách hoạt động:**
> Strategy này tách văn bản theo ranh giới câu rồi gom tối đa 3 câu vào một chunk. Với tài liệu y tế tiếng Việt, mỗi câu thường chứa một ý chẩn đoán/xử trí khá hoàn chỉnh, còn các bullet triệu chứng vẫn nằm gần nhau trong cùng vùng nội dung. Cách này tránh lỗi cắt ngang câu như fixed-size và tạo ít chunk hơn recursive trong nhiều tài liệu.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Các bài Vinmec có cấu trúc giải thích theo đoạn, tiêu đề phụ và danh sách triệu chứng; câu văn là đơn vị ngữ nghĩa tự nhiên hơn số ký tự. Benchmark cho thấy `SentenceChunker` đạt 10/10 giống fixed-size nhưng chỉ tạo 162 chunks trên toàn bộ 9 tài liệu, ít hơn fixed-size 212 chunks và recursive 282 chunks, nên cân bằng tốt giữa precision và chi phí lưu trữ.

**Code snippet (nếu custom):**
```python
# Không dùng custom chunker; sử dụng SentenceChunker có sẵn trong package src.
strategy = SentenceChunker(max_sentences_per_chunk=3)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 9 tài liệu Vinmec | FixedSizeChunker | 212 | khoảng 300 | 10/10, nhưng một số chunk có thể cắt ngang ý |
| 9 tài liệu Vinmec | RecursiveChunker | 282 | khoảng 190-200 | 9/10, Q3 relevant ở top-3 nhưng không ở top-1 |
| 9 tài liệu Vinmec | **SentenceChunker của tôi** | 162 | khoảng 250-320 | **10/10**, ít chunk hơn và đọc tự nhiên hơn |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker, 3 câu/chunk | 10/10 | Chunk dễ đọc, ít hơn fixed/recursive, giữ trọn câu | Có thể tạo chunk dài hơn `chunk_size` mong muốn nếu câu quá dài |
| Baseline A | FixedSizeChunker, 300 chars, overlap 50 | 10/10 | Ổn định, đơn giản, có overlap giữ ngữ cảnh | Có thể cắt ngang câu hoặc bullet |
| Baseline B | RecursiveChunker, 300 chars | 9/10 | Tôn trọng separator đoạn/dòng/câu | Tạo nhiều chunk nhỏ, Q3 top-1 bị nhiễu bởi tài liệu sốt rét |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `SentenceChunker` là lựa chọn tốt nhất trong thử nghiệm này vì đạt cùng điểm 10/10 với fixed-size nhưng dùng ít chunk hơn và giữ nội dung tự nhiên hơn. Với các bài tư vấn y tế dạng giải thích, một chunk gồm vài câu giúp giữ đủ triệu chứng/ngữ cảnh mà vẫn không quá dài.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em strip text đầu vào, dùng regex `(?<=[.!?])\s+` để tách tại khoảng trắng sau dấu kết thúc câu, rồi gom tối đa `max_sentences_per_chunk` câu vào một chunk. Với input rỗng trả về list rỗng; nếu không tách được câu thì giữ nguyên text thành một chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán thử separator theo thứ tự ưu tiên: đoạn văn, dòng, câu, khoảng trắng, rồi fallback cắt theo ký tự. Base case là text rỗng hoặc text đã ngắn hơn `chunk_size`; nếu một piece vẫn quá dài thì `_split` gọi đệ quy với separator tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi `Document` được chuẩn hóa thành record gồm `id`, `content`, `metadata`, và embedding; metadata có `doc_id` để truy vết nguồn, đồng thời giữ nguyên `doc_id` sẵn có khi document là chunk của một tài liệu gốc. Search embed query rồi tính dot product với từng record, sort giảm dần theo `score`, và trả về tối đa `top_k` kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc record theo metadata trước, sau đó mới chạy similarity search trên tập ứng viên đã lọc. `delete_document` xóa mọi record có `metadata["doc_id"]` trùng với `doc_id` cần xóa và trả về `True/False` tùy có xóa được gì không.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent retrieve top-k chunk từ store, tạo context block có thứ tự, source và score, rồi nhúng các block đó vào prompt. Prompt yêu cầu LLM trả lời dựa trên context và nói rõ nếu context chưa đủ.

### Test Results

```
# Paste output of: pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0 -- E:\Python\python.exe
rootdir: D:\AI Course\Day7\Day-07-Lab-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.18s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a high-level programming language. | Python is used to write software and scripts. | high | -0.0629 | Không, do mock embedding |
| 2 | Vector databases store embeddings for similarity search. | A vector store ranks text chunks by embedding similarity. | high | 0.0368 | Một phần |
| 3 | Customer support teams answer refund questions. | The recipe needs fresh basil and tomatoes. | low | -0.0074 | Có |
| 4 | Machine learning learns patterns from data. | Deep learning trains neural networks on data. | high | 0.0332 | Một phần |
| 5 | Dogs are loyal companion animals. | Cloud databases manage distributed storage. | low | 0.0420 | Không rõ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 5 bất ngờ nhất vì hai câu khác chủ đề nhưng điểm mock lại dương hơn một số cặp tương đồng về nghĩa. Điều này nhắc rằng `_mock_embed` chỉ phục vụ test deterministic, không thật sự hiểu ngữ nghĩa; khi dùng embedder thật như `all-MiniLM-L6-v2` hoặc OpenAI embeddings, kỳ vọng điểm similarity sẽ phản ánh ý nghĩa tốt hơn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Làm thế nào để phân biệt sốt xuất huyết và sốt virus? | Dựa vào xét nghiệm Dengue/công thức máu, dấu hiệu xuất huyết, giảm tiểu cầu, Hct tăng; sốt virus thường có triệu chứng hô hấp/tiêu hóa và tự khỏi khoảng 7 ngày. |
| 2 | Trẻ em sốt đến bao nhiêu độ thì cần uống thuốc hạ sốt? | Trẻ từ 38°C được xem là thật sự sốt; thuốc hạ sốt nên dùng khi trẻ sốt từ 38,5°C trở lên. Query này dùng filter `audience=tre_em`. |
| 3 | Triệu chứng của sốt xuất huyết nặng là gì? | Đau bụng dữ dội, nôn kéo dài, thở nhanh, chảy máu nướu/mũi, mệt/bồn chồn, có máu trong chất nôn hoặc phân, khát nhiều, da nhợt/lạnh, yếu. |
| 4 | Sốt rét và sốt xuất huyết khác nhau như thế nào? | Sốt rét do Plasmodium qua muỗi Anopheles, thường sốt theo cơn rét run/nóng/vã mồ hôi; sốt xuất huyết do Dengue qua muỗi Aedes, sốt cao liên tục và có nguy cơ xuất huyết. |
| 5 | Sốt phát ban khác sốt xuất huyết như thế nào? | Khi căng da, ban sốt phát ban mất đi rồi đỏ lại khi buông; nốt xuất huyết do sốt xuất huyết không mất sau khi căng da. Query này dùng filter `type=phan_biet`. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Phân biệt sốt xuất huyết và sốt virus | Chunk "Cách phân biệt sốt virus và sốt xuất huyết" từ `huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi` | 0.7556 | Có | Phân biệt bằng dịch tễ, test Dengue/công thức máu, dấu hiệu xuất huyết và diễn tiến bệnh. |
| 2 | Trẻ sốt bao nhiêu độ thì uống hạ sốt? | Chunk từ `tre-sot-den-dau-moi-phai-uong-thuoc-ha-sot-vi`, filter `audience=tre_em` | 0.6610 | Có | Trẻ sốt từ 38°C cần theo dõi; thuốc hạ sốt nên dùng khi từ 38,5°C trở lên. |
| 3 | Triệu chứng sốt xuất huyết nặng | Chunk "Triệu chứng sốt xuất huyết" từ `sot-xuat-huyet-va-sot-xuat-huyet-nang` | 0.7137 | Có | Nặng khi có đau bụng dữ dội, nôn kéo dài, thở nhanh, chảy máu, bồn chồn/mệt, da lạnh/nhợt. |
| 4 | Sốt rét khác sốt xuất huyết | Chunk mở đầu từ `phan-biet-sot-ret-va-sot-xuat-huyet-vi` | 0.7444 | Có | Khác ở tác nhân, muỗi truyền bệnh, thời gian ủ bệnh và kiểu sốt/triệu chứng đi kèm. |
| 5 | Sốt phát ban khác sốt xuất huyết | Chunk mở đầu từ `sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi`, filter `type=phan_biet` | 0.8664 | Có | Căng da là cách phân biệt nhanh: ban sốt phát ban mất màu, nốt xuất huyết thì không. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

### Kết Quả So Sánh 3 Strategy

| Strategy | Tổng chunks | Retrieval score | Nhận xét |
|----------|-------------|-----------------|----------|
| FixedSizeChunker | 212 | 10/10 | Top-3 tốt nhưng chunk có thể cắt ngang câu. |
| SentenceChunker | 162 | 10/10 | Ít chunk nhất trong nhóm đạt điểm tối đa; chọn làm strategy cá nhân. |
| RecursiveChunker | 282 | 9/10 | Nhiều chunk nhỏ; Q3 có relevant trong top-3 nhưng top-1 chưa đúng. |

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Trong phần so sánh strategy, em thấy fixed-size tuy đơn giản nhưng vẫn rất mạnh khi có overlap và embedding lexical tốt. Tuy nhiên, điểm số retrieval không phải toàn bộ câu chuyện: số lượng chunk và độ dễ đọc của chunk cũng quan trọng khi agent cần dùng context để trả lời.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Metadata filtering là cách rất hiệu quả để giảm nhiễu khi nhiều tài liệu dùng chung từ khóa "sốt". Query về trẻ em nhờ filter `audience=tre_em` chỉ tìm trong đúng tài liệu dành cho trẻ, giúp top-3 sạch hơn nhiều.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Em sẽ chuẩn hóa `disease_type` thành list thay vì string có dấu phẩy để filter được linh hoạt hơn, ví dụ một tài liệu vừa thuộc `sot_virus` vừa thuộc `sot_xuat_huyet`. Em cũng sẽ thêm metadata `section_title` cho từng chunk để biết kết quả đến từ phần "Triệu chứng", "Điều trị" hay "Phân biệt".

**Failure case phân tích:**
> Với `RecursiveChunker`, query Q3 về triệu chứng sốt xuất huyết nặng chỉ đạt 1/2 vì top-1 trả về tài liệu `phan-biet-sot-ret-va-sot-xuat-huyet-vi`, còn tài liệu đúng nằm trong top-3. Nguyên nhân là nhiều chunk nhỏ cùng chứa các từ khóa chung như "sốt", "xuất huyết", "triệu chứng", làm lexical embedding bị nhiễu. Cải thiện: thêm filter `disease_type=sot_xuat_huyet`, tăng kích thước chunk cho phần triệu chứng, hoặc lưu `section_title` để ưu tiên đoạn có tiêu đề liên quan.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **chưa tính Demo** |
