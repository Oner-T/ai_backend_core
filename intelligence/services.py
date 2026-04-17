import os
import re
import json
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pgvector.django import CosineDistance
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from sentence_transformers import CrossEncoder
from langsmith import traceable
from intelligence.models import DocumentChunk, DocumentSection

_cross_encoder: CrossEncoder | None = None
_embeddings_model: HuggingFaceEmbeddings | None = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return _cross_encoder

def get_embeddings_model() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        # 2.24GB model — load once per process, reuse for all requests
        _embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return _embeddings_model

CROSS_REF_PATTERN = re.compile(r'\b(\d{1,3})\s+(?:inci|nci|ncı|üncü|ıncı|uncu|üncu)\b', re.IGNORECASE)
VALID_KVKK_MADDE_RANGE = range(1, 34)  # KVKK has articles 1–33

TURKISH_ORDINAL_TO_INT = {
    "BİRİNCİ": 1,
    "İKİNCİ": 2,
    "ÜÇÜNCÜ": 3,
    "DÖRDÜNCÜ": 4,
    "BEŞİNCİ": 5,
    "ALTINCI": 6,
    "YEDİNCİ": 7,
    "SEKİZİNCİ": 8,
    "DOKUZUNCU": 9,
    "ONUNCU": 10,
    "ONBİRİNCİ": 11,
    "ONİKİNCİ": 12,
}

def bolum_name_to_int(bolum_text):
    ordinal = bolum_text.replace("BÖLÜM", "").strip()
    return TURKISH_ORDINAL_TO_INT.get(ordinal)

def ingest_kvkk_document():
    print("🧹 Clearing old data from Supabase...")
    DocumentChunk.objects.all().delete()
    DocumentSection.objects.all().delete()

    file_path = os.path.join(settings.BASE_DIR, 'KVKK.pdf')
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        return

    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Patterns
    bolum_pattern    = re.compile(r"^([A-ZÇĞİÖŞÜ]+\sBÖLÜM)")
    madde_pattern    = re.compile(r"^MADDE\s(\d{1,3})")
    num_para_pattern = re.compile(r"^\((\d+)\)")                        # (2), (3)... within a madde
    footnote_pattern = re.compile(r"^\d+\s+\d{1,2}/\d{1,2}/\d{4}")    # e.g. "2 2/7/2018 tarihli..."
    # Definition-article lettered items: "a) Term: explanation" — split only in Madde 3 (Tanımlar)
    def_item_pattern = re.compile(r"^([a-zçğışöü])\)\s+\S")

    # State
    current_section_obj  = None
    current_madde_num    = None
    current_madde_title  = None
    last_seen_line       = None   # used for title detection
    paragraph_buffer     = []
    chunk_objects        = []
    global_chunk_index   = 1

    # Flatten all pages into one line stream
    all_lines = []
    for page in pages:
        all_lines.extend(page.page_content.split('\n'))

    def is_title_candidate(line: str) -> bool:
        """A title line is short, has no ending punctuation, and isn't a list item or numbered para."""
        return (
            len(line) < 100
            and not line.endswith(('.', ',', ';', ':'))
            and not line.startswith('(')
            and not num_para_pattern.match(line)
            and not madde_pattern.match(line)
            and not bolum_pattern.match(line)
        )

    def flush_chunk():
        nonlocal global_chunk_index, paragraph_buffer
        if not paragraph_buffer or current_section_obj is None or current_madde_num is None:
            paragraph_buffer = []
            return
        raw = " ".join(paragraph_buffer).strip()
        if len(raw) < 15:
            paragraph_buffer = []
            return
        title_part = f" — {current_madde_title}" if current_madde_title else ""
        header     = f"[Bölüm {current_section_obj.number}, Madde {current_madde_num}{title_part}]"
        chunk_objects.append(DocumentChunk(
            section       = current_section_obj,
            madde         = current_madde_num,
            madde_title   = current_madde_title,
            document_name = "KVKK.pdf",
            chunk_index   = global_chunk_index,
            content       = f"{header}\n{raw}",
        ))
        global_chunk_index += 1
        paragraph_buffer = []

    print("✂️ Parsing structure and creating chunks...")

    for line in all_lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        if footnote_pattern.match(clean_line):
            continue

        # ── New BÖLÜM ──────────────────────────────────────────────────────
        bolum_match = bolum_pattern.match(clean_line)
        if bolum_match:
            flush_chunk()
            current_madde_num   = None   # reset so inter-section lines don't pollute
            current_madde_title = None
            last_seen_line      = None
            bolum_name   = bolum_match.group(1)
            bolum_number = bolum_name_to_int(bolum_name)
            current_section_obj, _ = DocumentSection.objects.get_or_create(number=bolum_number)
            print(f"📍 Found Section: {bolum_name} → {bolum_number}")
            continue

        # ── New MADDE ───────────────────────────────────────────────────────
        madde_match = madde_pattern.match(clean_line)
        if madde_match:
            # The line immediately before this MADDE is the article title if it
            # looks like one. Remove it from the buffer if it was added there.
            candidate_title = None
            if last_seen_line and is_title_candidate(last_seen_line):
                candidate_title = last_seen_line
                if paragraph_buffer and paragraph_buffer[-1] == last_seen_line:
                    paragraph_buffer.pop()

            flush_chunk()
            current_madde_num   = int(madde_match.group(1))
            current_madde_title = candidate_title
            last_seen_line      = clean_line
            paragraph_buffer.append(clean_line)
            print(f"   ↳ MADDE {current_madde_num} — {current_madde_title}")
            continue

        # ── Numbered paragraph (2), (3)… → chunk boundary ──────────────────
        num_match = num_para_pattern.match(clean_line)
        if num_match and current_madde_num is not None:
            para_num = int(num_match.group(1))
            if para_num > 1:          # (1) is inline with MADDE line; (2)+ split
                flush_chunk()
                paragraph_buffer.append(clean_line)
                last_seen_line = clean_line
                continue

        # ── Madde 3 definition items: a), b), c)… → each is its own chunk ────
        if current_madde_num == 3 and def_item_pattern.match(clean_line):
            flush_chunk()
            paragraph_buffer.append(clean_line)
            last_seen_line = clean_line
            continue

        # ── Regular content line ────────────────────────────────────────────
        if current_madde_num is not None:
            paragraph_buffer.append(clean_line)

        last_seen_line = clean_line

    flush_chunk()   # save the very last paragraph

    print(f"🧠 Generating {len(chunk_objects)} embeddings with multilingual-e5-large...")
    from sentence_transformers import SentenceTransformer as _ST
    _ingest_model = _ST("intfloat/multilingual-e5-large")

    # Ablation test (Config C vs D/E) proved e5-large + WITH HEADERS = MRR 0.904
    # vs content-only = MRR 0.777. e5-large reads headers as structured metadata,
    # which anchors each chunk's topic and improves retrieval significantly.
    content_for_embedding = [f"passage: {c.content}" for c in chunk_objects]
    vectors = _ingest_model.encode(content_for_embedding, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    for chunk, vector in zip(chunk_objects, vectors):
        chunk.embedding = vector.tolist()

    print("💾 Saving to Supabase...")
    DocumentChunk.objects.bulk_create(chunk_objects)
    print(f"✅ Done — {len(chunk_objects)} chunks saved.")


@traceable(name="resolve_cross_references")
def resolve_cross_references(chunks: list, already_fetched_maddes: set) -> list:
    """Scan retrieved chunks for ordinal references to other KVKK articles and fetch them."""
    referenced_maddes = set()
    for chunk in chunks:
        for match in CROSS_REF_PATTERN.finditer(chunk.content):
            num = int(match.group(1))
            if num in VALID_KVKK_MADDE_RANGE and num not in already_fetched_maddes:
                referenced_maddes.add(num)

    if not referenced_maddes:
        return []

    extra_chunks = list(
        DocumentChunk.objects.select_related('section')
        .filter(madde__in=referenced_maddes)
        .order_by('madde', 'chunk_index')
    )
    return extra_chunks


def _log_tokens(label: str, response) -> dict:
    """Extract and print token usage from an Ollama response."""
    usage = response.usage_metadata or {}
    inp  = usage.get("input_tokens", 0)
    out  = usage.get("output_tokens", 0)
    total = usage.get("total_tokens", inp + out)
    return {"input": inp, "output": out, "total": total}


MADDE_REGEX  = re.compile(r'\bMadde\s+(\d{1,3})\b', re.IGNORECASE)
BOLUM_REGEX  = re.compile(r'\b(\d{1,2})\s*\.?\s*[Bb]ölüm\b|\b[Bb]ölüm\s+(\d{1,2})\b')


def _extract_filters_by_regex(text: str) -> dict:
    """Reliably extract explicit madde/bolum numbers using regex — no LLM needed."""
    madde_match = MADDE_REGEX.search(text)
    bolum_match = BOLUM_REGEX.search(text)
    madde = int(madde_match.group(1)) if madde_match else None
    bolum = int(bolum_match.group(1) or bolum_match.group(2)) if bolum_match else None
    return {"madde": madde, "bolum": bolum}


@traceable(name="analyze_query")
def analyze_query(question: str, history: list) -> dict:
    """Regex-only filter extraction. No LLM call — rewrite was garbling Turkish and is not used for embedding."""
    filters = _extract_filters_by_regex(question)
    return {"madde": filters["madde"], "bolum": filters["bolum"]}


def bm25_search(query: str, queryset, top_k: int = 10) -> list:
    """PostgreSQL full-text search as a BM25 approximation."""
    search_query = SearchQuery(query, config='simple')
    results = (
        queryset
        .annotate(bm25_rank=SearchRank(SearchVector('content', config='simple'), search_query))
        .filter(bm25_rank__gt=0)
        .order_by('-bm25_rank')[:top_k]
    )
    return list(results)


@traceable(name="bm25_search")
def bm25_search_traced(query: str, chunks: list) -> dict:
    """LangSmith-visible wrapper: returns ranked list with scores."""
    return {
        "total": len(chunks),
        "ranked": [
            {
                "rank": i + 1,
                "madde": c.madde,
                "bolum": c.section.number if c.section else None,
                "madde_title": c.madde_title,
                "bm25_score": round(float(getattr(c, 'bm25_rank', 0)), 4),
                "preview": c.content[:120],
            }
            for i, c in enumerate(chunks)
        ],
    }


def reciprocal_rank_fusion(vector_chunks: list, bm25_chunks: list, top_k: int = 8, k: int = 60) -> list:
    """Merge two ranked lists using Reciprocal Rank Fusion. Higher score = better."""
    scores: dict[int, float] = {}
    chunk_map: dict[int, object] = {}

    for rank, chunk in enumerate(vector_chunks):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(bm25_chunks):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    vector_ids  = {c.id for c in vector_chunks}
    bm25_ids    = {c.id for c in bm25_chunks}
    overlap     = len(vector_ids & bm25_ids)
    bm25_unique = len(bm25_ids - vector_ids)

    return [chunk_map[cid] for cid in sorted_ids]


@traceable(name="retrieval_pipeline")
def retrieval_pipeline_traced(question: str, vector_chunks: list, bm25_chunks: list, fused_chunks: list) -> dict:
    """LangSmith-visible summary of all three retrieval stages."""
    vector_ids = {c.id for c in vector_chunks}
    bm25_ids   = {c.id for c in bm25_chunks}

    def serialize(chunks, source_tag):
        return [
            {
                "rank": i + 1,
                "source": source_tag,
                "madde": c.madde,
                "bolum": c.section.number if c.section else None,
                "madde_title": c.madde_title,
                "in_vector": c.id in vector_ids,
                "in_bm25":   c.id in bm25_ids,
                "preview": c.content[:120],
            }
            for i, c in enumerate(chunks)
        ]

    return {
        "question": question,
        "vector_results":  serialize(vector_chunks, "vector"),
        "bm25_results":    serialize(bm25_chunks,   "bm25"),
        "fused_results":   serialize(fused_chunks,  "rrf"),
        "stats": {
            "vector_count":  len(vector_chunks),
            "bm25_count":    len(bm25_chunks),
            "overlap":       len(vector_ids & bm25_ids),
            "bm25_unique":   len(bm25_ids - vector_ids),
            "vector_unique": len(vector_ids - bm25_ids),
            "fused_count":   len(fused_chunks),
        },
    }


@traceable(name="rerank_chunks", metadata={"step": "cross_encoder_rerank"})
def rerank_chunks(query: str, chunks: list, top_k: int = 5) -> list:
    """Re-score retrieved chunks with a cross-encoder and return the top_k most relevant."""
    if not chunks:
        return chunks
    cross_encoder = get_cross_encoder()
    pairs = [(query, chunk.content) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    # Log ordering for LangSmith visibility
    for rank, (score, chunk) in enumerate(ranked[:top_k]):
        print(f"  #{rank+1} Madde {chunk.madde} (score={score:.3f})")

    top = [chunk for _, chunk in ranked[:top_k]]
    return top


@traceable(name="query_kvkk")
def query_kvkk(question: str, history: list | None = None) -> dict:
    if history is None:
        history = []
    # Keep last 6 messages (3 turns) to avoid bloating the context
    history = history[-6:]

    # 1. Single LLM call: extract filters + rewrite query
    analysis = analyze_query(question, history)
    filters = {"madde": analysis["madde"], "bolum": analysis["bolum"]}

    # 2. Embed the ORIGINAL question — E5 requires "query: " prefix at query time.
    # We do NOT embed the rewritten query: llama3.2 garbles Turkish text and the
    # rewrite corrupts retrieval. The rewrite is kept only for LLM context resolution
    # in follow-up questions (pronoun resolution) but is not used for vector search.
    question_embedding = get_embeddings_model().embed_query("query: " + question)

    # 3. Build queryset — apply metadata filters if the user referenced a specific madde/bolum
    # Exclude Mülga (deleted/repealed) paragraphs — their headers match semantically but content is empty/useless
    queryset = DocumentChunk.objects.select_related('section').exclude(content__icontains='Mülga')
    if filters["madde"] is not None:
        queryset = queryset.filter(madde=filters["madde"])

    if filters["bolum"] is not None:
        queryset = queryset.filter(section__number=filters["bolum"])


    # 4. Retrieve strategy:
    #    - Specific madde referenced → fetch ALL chunks for that madde in document order
    #    - General question → vector search top-8 (PHASE 2: BM25+RRF disabled — hurts Turkish queries)
    if filters["madde"] is not None:
        similar_chunks = list(queryset.order_by('chunk_index'))
    else:
        similar_chunks = list(
            queryset
            .annotate(distance=CosineDistance('embedding', question_embedding))
            .order_by('distance')[:8]
        )

        # Send structured retrieval data to LangSmith
        retrieval_pipeline_traced(question, similar_chunks, [], similar_chunks)

        # PHASE 2 (disabled): BM25 + RRF — config='simple' hurts Turkish morphology
        # bm25_results = bm25_search(rewritten_question, queryset, top_k=10)
        # similar_chunks = reciprocal_rank_fusion(similar_chunks, bm25_results, top_k=8)

        # PHASE 3 (disabled): cross-encoder reranking
        # similar_chunks = rerank_chunks(question, similar_chunks, top_k=5)

    # 5. Out-of-scope check — if best vector match is too far away
    if not similar_chunks:
        return {
            "answer": "Bu soru KVKK kapsamında değil veya belgede bu konuda bilgi bulunmuyor.",
            "sources": []
        }
    if filters["madde"] is None:
        best_vector_distance = getattr(similar_chunks[0], 'distance', 0)
        if best_vector_distance > 0.6:
            return {
                "answer": "Bu soru KVKK kapsamında değil veya belgede bu konuda bilgi bulunmuyor.",
                "sources": []
            }

    # PHASE 4 (disabled): cross-reference resolution
    # primary_maddes = {c.madde for c in similar_chunks if c.madde is not None}
    # referenced_chunks = resolve_cross_references(similar_chunks, already_fetched_maddes=primary_maddes)
    referenced_chunks = []

    # 6. Build context
    context_parts = []
    for chunk in similar_chunks:
        bolum = chunk.section.number if chunk.section else "?"
        madde = chunk.madde or "?"
        context_parts.append(f"[Bölüm {bolum}, Madde {madde}]\n{chunk.content}")

    context = "\n\n".join(context_parts)

    # 5. Call Gemini for answer generation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=60,
    )

    history_messages = []
    for m in history:
        if m["role"] == "user":
            history_messages.append(HumanMessage(content=m["content"]))
        else:
            history_messages.append(SystemMessage(content=f"[Your previous answer]: {m['content']}"))

    messages = [
        SystemMessage(content=(
            "You are an expert on KVKK (Kişisel Verilerin Korunması Kanunu — Turkish Personal Data Protection Law). "
            "Answer the user's question strictly based on the provided KVKK context passages. "
            "If the answer is not present in the context, say so clearly — do not speculate or use outside knowledge. "
            "Always cite the Bölüm (Chapter) and Madde (Article) your answer is based on. "
            "Reply in the same language the user used (Turkish question → Turkish answer, English question → English answer). "
            "When user asks for all information about a specific topic, give all relevant passages from the context that mention that topic, along with their citations. "
            "Use the conversation history below to understand follow-up questions and references to previous answers.\n\n"
            "Here are examples of ideal answers:\n\n"
            "Example 1:\n"
            "Question: Kişisel veri nedir?\n"
            "Answer: Kişisel veri, kimliği belirli veya belirlenebilir gerçek kişiye ilişkin her türlü bilgidir. "
            "(Bölüm 1, Madde 3)\n\n"
            "Example 2:\n"
            "Question: Madde 5 kapsamında kişisel veri işleme şartları nelerdir?\n"
            "Answer: Madde 5'e göre kişisel veriler kural olarak ilgili kişinin açık rızası olmaksızın işlenemez. "
            "Ancak aşağıdaki hallerden birinin varlığı hâlinde açık rıza aranmaz:\n"
            "- Kanunlarda açıkça öngörülmesi\n"
            "- Fiili imkânsızlık nedeniyle rızasını açıklayamayacak durumda olan kişinin kendisinin ya da başkasının hayatı veya beden bütünlüğünün korunması\n"
            "- Bir sözleşmenin kurulması veya ifasıyla doğrudan ilgili olması\n"
            "- Veri sorumlusunun hukuki yükümlülüğünü yerine getirebilmesi\n"
            "- İlgili kişinin kendisi tarafından alenileştirilmiş olması\n"
            "- Bir hakkın tesisi, kullanılması veya korunması için zorunlu olması\n"
            "- İlgili kişinin temel hak ve özgürlüklerine zarar vermemek kaydıyla veri sorumlusunun meşru menfaatleri için zorunlu olması\n"
            "(Bölüm 2, Madde 5)\n\n"
            "Example 3:\n"
            "Question: İlgili kişinin hakları nelerdir?\n"
            "Answer: İlgili kişi Madde 11 kapsamında aşağıdaki haklara sahiptir:\n"
            "- Kişisel verilerinin işlenip işlenmediğini öğrenme\n"
            "- İşlenmişse buna ilişkin bilgi talep etme\n"
            "- İşlenme amacını ve amacına uygun kullanılıp kullanılmadığını öğrenme\n"
            "- Yurt içinde veya yurt dışında aktarıldığı üçüncü kişileri bilme\n"
            "- Eksik veya yanlış işlenmiş olması hâlinde bunların düzeltilmesini isteme\n"
            "- Kanunda öngörülen şartlar çerçevesinde silinmesini veya yok edilmesini isteme\n"
            "- Düzeltme, silme ve yok etme işlemlerinin üçüncü kişilere bildirilmesini isteme\n"
            "- İşlenen verilerin münhasıran otomatik sistemler vasıtasıyla analiz edilmesi suretiyle aleyhine bir sonucun ortaya çıkmasına itiraz etme\n"
            "- Kanuna aykırı işlenmesi sebebiyle zarara uğraması hâlinde zararın giderilmesini talep etme\n"
            "(Bölüm 3, Madde 11)"
        )),
        *history_messages,
        HumanMessage(content=f"KVKK Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    _log_tokens("answer", response)

    primary_ids = {c.id for c in similar_chunks}
    all_source_chunks = similar_chunks + referenced_chunks
    # Deduplicate by (bolum, madde) and sort in document order
    seen = set()
    ordered_sources = []
    for c in sorted(all_source_chunks, key=lambda x: (x.section.number if x.section else 99, x.madde or 99)):
        key = (c.section.number if c.section else None, c.madde)
        if key not in seen:
            seen.add(key)
            ordered_sources.append(c)

    return {
        "answer": response.content,
        "sources": [
            {
                "bolum": c.section.number if c.section else None,
                "madde": c.madde,
                "madde_title": c.madde_title,
                "is_cross_reference": c.id not in primary_ids,
                # Strip the "[Bölüm X, Madde Y]" header line prepended during ingestion
                "content": c.content.split('\n', 1)[1].strip() if '\n' in c.content else c.content,
            }
            for c in ordered_sources
        ],
    }