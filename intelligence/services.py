import os
import re
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pgvector.django import CosineDistance
from langsmith import traceable
from intelligence.models import DocumentChunk, DocumentSection

ANSWER_MODEL = "gemini-2.5-flash"

# ── Embeddings: local model in dev, HuggingFace Inference API in production ───

_USE_API = bool(os.getenv("HF_TOKEN")) and not bool(os.getenv("USE_LOCAL_EMBEDDINGS"))

if _USE_API:
    from huggingface_hub import InferenceClient as _HFClient
    _hf_client: "_HFClient | None" = None

    def _get_hf_client() -> "_HFClient":
        global _hf_client
        if _hf_client is None:
            _hf_client = _HFClient(token=os.getenv("HF_TOKEN"))
        return _hf_client

    def embed_query(text: str) -> list[float]:
        result = _get_hf_client().feature_extraction(
            text, model="intfloat/multilingual-e5-large"
        )
        import numpy as np
        arr = np.array(result)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()

else:
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmbeddings
    _local_model: "_HFEmbeddings | None" = None

    def _get_local_model() -> "_HFEmbeddings":
        global _local_model
        if _local_model is None:
            _local_model = _HFEmbeddings(model_name="intfloat/multilingual-e5-large")
        return _local_model

    def embed_query(text: str) -> list[float]:
        return _get_local_model().embed_query(text)


# ── Document registries ──────────────────────────────────────────────────────

TR_DOCUMENTS = [
    {"file": "files/tr/KVKK.pdf",                                                                              "name": "KVKK",                   "short": "KVKK"},
    {"file": "files/tr/KVKK_son.pdf",                                                                          "name": "Yurt Dışına Aktarım Yönetmeliği", "short": "Aktarım Yönetmeliği"},
    {"file": "files/tr/SilmeYonetmelik_KVKK.pdf",                                                             "name": "Silme Yönetmeliği",       "short": "Silme Yönetmeliği"},
    {"file": "files/tr/AydınlatmaYükümlülüğününYerineGetirilmesindeUyulacakUsulveEsaslarHakkındaTebliğ_KVKK.pdf", "name": "Aydınlatma Tebliği",      "short": "Aydınlatma Tebliği"},
    {"file": "files/tr/VeriSorumlularıSiciliHakkındaYönetmelik_KVKK.pdf",                                     "name": "VERBİS Yönetmeliği",      "short": "VERBİS Yönetmeliği"},
]

EU_DOCUMENTS = [
    {"file": "files/eu/CELEX_32016R0679_EN_TXT.pdf", "name": "GDPR",               "short": "GDPR"},
    {"file": "files/eu/CELEX_32002L0058_EN_TXT.pdf", "name": "ePrivacy Directive", "short": "ePrivacy"},
    {"file": "files/eu/OJ_L_202401689_EN_TXT.pdf",   "name": "EU AI Act",          "short": "AI Act"},
]

DOCUMENTS = TR_DOCUMENTS  # backward-compat alias


# ── Turkish parser helpers ───────────────────────────────────────────────────

_PRIVATE_DIGIT_MAP = str.maketrans({
    '': '0', '': '1', '': '2', '': '3',
    '': '4', '': '5', '': '6', '': '7',
    '': '8', '': '9',
    '\xa0': ' ',
})

def _normalize(line: str) -> str:
    return line.translate(_PRIVATE_DIGIT_MAP)

_WEB_NOISE = re.compile(
    r'^(TR|EN|DE)$'
    r'|^\s*/'
    r'|^Yayınlanma Tarihi'
    r'|^M\s*ENÜ\s*$'
    r'|^\d{1,2}/\d{2}/\d{2},\s*\d'
    r'|https?://'
)

_FOOTER_STOP = re.compile(
    r'〉'
    r'|^Yönetmelikte Değişiklik Yapan'
    r'|KİŞİSEL VERİLERİ KORUMA KURUMU'
)

TURKISH_ORDINAL_TO_INT = {
    "BİRİNCİ": 1, "İKİNCİ": 2, "ÜÇÜNCÜ": 3, "DÖRDÜNCÜ": 4,
    "BEŞİNCİ": 5, "ALTINCI": 6, "YEDİNCİ": 7, "SEKİZİNCİ": 8,
    "DOKUZUNCU": 9, "ONUNCU": 10, "ONBİRİNCİ": 11, "ONİKİNCİ": 12,
}

def bolum_name_to_int(bolum_text: str) -> int | None:
    ordinal = bolum_text.replace("BÖLÜM", "").strip()
    return TURKISH_ORDINAL_TO_INT.get(ordinal)


# ── EU parser helpers ────────────────────────────────────────────────────────

ROMAN_TO_INT = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14,
}

_EU_NOISE = re.compile(
    r'^\d{1,2}\.\d{1,2}\.\d{4}\b'    # date "4.7.2016" or "12.7.2024"
    r'|^Official Journal'              # OJ page header
    r'|^L\s+\d'                       # "L 119" series marker
    r'|^\d+/\d+\s+ELI:'              # "44/144 ELI: http://..."
    r'|^ELI:'                         # bare ELI URI
    r'|^\d{4}/\d+$'                   # "2024/1689"
    r'|^C\s+\d'                       # "C 123" series
    r'|^OJ\s+L,'                      # "OJ L, 12.7.2024"
    r'|^HA\s*VE\s+ADOPTED'           # "HAVE ADOPTED THIS REGULATION:"
)

# Both PDFs have OCR-split words: "Ar ticle" instead of "Article",
# "C HAPTER" instead of "CHAPTER" — the regexes handle both forms.
_EU_ARTICLE = re.compile(r'^Ar\s*ticle\s+(\d{1,3})\s*$')
_EU_CHAPTER = re.compile(r'^C\s*HAPTER\s+([IVXLCDM]+)\s*$')
_EU_SECTION = re.compile(r'^Section\s+\d+\s*$')
_EU_ANNEX   = re.compile(r'^ANNEX\s*\w*\s*$')
_EU_NUM_PARA = re.compile(r'^(\d+)\.\s')


# ── Turkish document parser ──────────────────────────────────────────────────

def _parse_single_document(
    file_path: str,
    document_name: str,
    doc_short_name: str,
    global_chunk_index: int,
) -> tuple[list, int]:
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return [], global_chunk_index

    print(f"\n📄 Loading: {file_path}")
    pages = PyPDFLoader(file_path).load()

    bolum_pattern    = re.compile(r"^([A-ZÇĞİÖŞÜ]+\s+BÖLÜM)$|^(BÖLÜM)$")
    madde_pattern    = re.compile(r"^MADDE\s+(\d{1,3})")
    madde_no_num     = re.compile(r"^MADDE\s*[\-–]")
    num_para_pattern = re.compile(r"^\((\d+)\)")
    footnote_pattern = re.compile(r"^\d+\s+\d{1,2}/\d{1,2}/\d{4}")
    def_item_pattern = re.compile(r"^([a-zçğışöü])\)\s+\S")

    current_section_obj  = None
    current_madde_num    = None
    current_madde_title  = None
    last_seen_line       = None
    paragraph_buffer: list[str] = []
    chunk_objects: list[DocumentChunk] = []
    madde_seq            = 0
    stop_parsing         = False

    all_lines = []
    for page in pages:
        all_lines.extend(page.page_content.split('\n'))

    def is_title_candidate(line: str) -> bool:
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
        if not paragraph_buffer or current_madde_num is None:
            paragraph_buffer = []
            return
        raw = " ".join(paragraph_buffer).strip()
        if len(raw) < 15:
            paragraph_buffer = []
            return
        title_part = f" — {current_madde_title}" if current_madde_title else ""
        if current_section_obj is not None:
            header = f"[{doc_short_name}, Bölüm {current_section_obj.number}, Madde {current_madde_num}{title_part}]"
        else:
            header = f"[{doc_short_name}, Madde {current_madde_num}{title_part}]"
        chunk_objects.append(DocumentChunk(
            section       = current_section_obj,
            madde         = current_madde_num,
            madde_title   = current_madde_title,
            document_name = document_name,
            chunk_index   = global_chunk_index,
            content       = f"{header}\n{raw}",
        ))
        global_chunk_index += 1
        paragraph_buffer = []

    print(f"  ✂️  Parsing structure...")

    for line in all_lines:
        if stop_parsing:
            break

        clean_line = _normalize(line).strip()
        if not clean_line:
            continue

        if _FOOTER_STOP.search(clean_line):
            flush_chunk()
            stop_parsing = True
            break

        if _WEB_NOISE.match(clean_line):
            continue
        if footnote_pattern.match(clean_line):
            continue

        bolum_match = bolum_pattern.match(clean_line)
        if bolum_match:
            flush_chunk()
            current_madde_num   = None
            current_madde_title = None

            if bolum_match.group(1):
                bolum_name = bolum_match.group(1)
            else:
                prev = (last_seen_line or "").strip().upper()
                if prev in TURKISH_ORDINAL_TO_INT:
                    bolum_name = prev + " BÖLÜM"
                else:
                    last_seen_line = clean_line
                    continue

            bolum_number = bolum_name_to_int(bolum_name)
            if bolum_number:
                current_section_obj, _ = DocumentSection.objects.get_or_create(number=bolum_number)
                print(f"    📍 {bolum_name} → {bolum_number}")
            last_seen_line = None
            continue

        madde_match    = madde_pattern.match(clean_line)
        madde_no_match = madde_no_num.match(clean_line) if not madde_match else None

        if madde_match or madde_no_match:
            candidate_title = None
            if last_seen_line and is_title_candidate(last_seen_line) and not _WEB_NOISE.match(last_seen_line):
                candidate_title = last_seen_line
                if paragraph_buffer and paragraph_buffer[-1] == last_seen_line:
                    paragraph_buffer.pop()

            flush_chunk()

            if madde_match:
                current_madde_num = int(madde_match.group(1))
            else:
                madde_seq += 1
                current_madde_num = madde_seq

            current_madde_title = candidate_title
            last_seen_line      = clean_line
            paragraph_buffer.append(clean_line)
            print(f"      ↳ MADDE {current_madde_num} — {current_madde_title}")
            continue

        num_match = num_para_pattern.match(clean_line)
        if num_match and current_madde_num is not None:
            para_num = int(num_match.group(1))
            if para_num > 1:
                flush_chunk()
                paragraph_buffer.append(clean_line)
                last_seen_line = clean_line
                continue

        is_tanim = (
            current_madde_title is not None
            and 'anım' in current_madde_title
        )
        if is_tanim and def_item_pattern.match(clean_line):
            flush_chunk()
            paragraph_buffer.append(clean_line)
            last_seen_line = clean_line
            continue

        if current_madde_num is not None:
            paragraph_buffer.append(clean_line)

        last_seen_line = clean_line

    flush_chunk()
    print(f"  ✅ {len(chunk_objects)} chunks parsed from {document_name}")
    return chunk_objects, global_chunk_index


# ── EU document parser ───────────────────────────────────────────────────────

def _parse_eu_document(
    file_path: str,
    document_name: str,
    doc_short_name: str,
    global_chunk_index: int,
) -> tuple[list, int]:
    """Parse EUR-Lex PDF (GDPR / ePrivacy / AI Act) into DocumentChunk objects."""
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return [], global_chunk_index

    print(f"\n📄 Loading: {file_path}")
    pages = PyPDFLoader(file_path).load()

    current_chapter_num    = None
    current_article_num    = None
    current_article_title  = None
    paragraph_buffer: list[str] = []
    chunk_objects: list[DocumentChunk] = []
    next_line_is_title     = False
    in_annex               = False

    all_lines = []
    for page in pages:
        all_lines.extend(page.page_content.split('\n'))

    def flush_eu_chunk():
        nonlocal global_chunk_index, paragraph_buffer
        if not paragraph_buffer or current_article_num is None:
            paragraph_buffer = []
            return
        raw = " ".join(paragraph_buffer).strip()
        if len(raw) < 20:
            paragraph_buffer = []
            return
        title_part = f" — {current_article_title}" if current_article_title else ""
        if current_chapter_num is not None:
            header = f"[{doc_short_name}, Chapter {current_chapter_num}, Article {current_article_num}{title_part}]"
        else:
            header = f"[{doc_short_name}, Article {current_article_num}{title_part}]"
        chunk_objects.append(DocumentChunk(
            section       = None,
            madde         = current_article_num,
            madde_title   = current_article_title,
            document_name = document_name,
            chunk_index   = global_chunk_index,
            content       = f"{header}\n{raw}",
        ))
        global_chunk_index += 1
        paragraph_buffer = []

    print(f"  ✂️  Parsing EU structure...")

    for line in all_lines:
        clean = line.strip()
        if not clean:
            continue

        # Skip EUR-Lex boilerplate
        if _EU_NOISE.match(clean):
            continue
        # Skip bare page numbers and standalone "EN"
        if re.match(r'^\d+$', clean) or clean == 'EN':
            continue

        # Annexes come after the operative articles — include but mark them
        if _EU_ANNEX.match(clean):
            flush_eu_chunk()
            in_annex = True
            current_article_num   = None
            current_article_title = None
            next_line_is_title    = False
            print(f"    📎 {clean}")
            continue

        # Chapter header
        chapter_match = _EU_CHAPTER.match(clean)
        if chapter_match:
            flush_eu_chunk()
            roman = chapter_match.group(1).upper()
            current_chapter_num = ROMAN_TO_INT.get(roman, current_chapter_num)
            next_line_is_title  = False
            print(f"    📍 Chapter {roman} → {current_chapter_num}")
            continue

        # Section subheader — just a structural marker, no state change needed
        if _EU_SECTION.match(clean):
            continue

        # Article header
        article_match = _EU_ARTICLE.match(clean)
        if article_match:
            flush_eu_chunk()
            current_article_num   = int(article_match.group(1))
            current_article_title = None
            next_line_is_title    = True
            in_annex              = False
            print(f"      ↳ Article {current_article_num}")
            continue

        # Title line — the first non-structural line after "Article N"
        if next_line_is_title:
            next_line_is_title = False
            # Accept as title if it's short, not all-caps noise, and not a numbered paragraph
            if (not _EU_NUM_PARA.match(clean)
                    and not _EU_CHAPTER.match(clean)
                    and not _EU_ARTICLE.match(clean)
                    and len(clean) < 150):
                current_article_title = clean
                print(f"         Title: {clean[:70]}")
                continue
            # else fall through and treat as content

        # Numbered paragraph → chunk boundary if para > 1
        num_match = _EU_NUM_PARA.match(clean)
        if num_match and current_article_num is not None:
            para_num = int(num_match.group(1))
            if para_num > 1:
                flush_eu_chunk()
            paragraph_buffer.append(clean)
            continue

        # Regular content
        if current_article_num is not None:
            paragraph_buffer.append(clean)

    flush_eu_chunk()
    print(f"  ✅ {len(chunk_objects)} chunks parsed from {document_name}")
    return chunk_objects, global_chunk_index


# ── Ingest ───────────────────────────────────────────────────────────────────

def ingest_all_documents(regime: str = 'all'):
    """Re-ingest documents. regime='all' clears everything; 'tr'/'eu' updates only that regime."""
    if regime == 'all':
        print("🧹 Clearing all existing data...")
        DocumentChunk.objects.all().delete()
        DocumentSection.objects.all().delete()
    elif regime == 'tr':
        print("🧹 Clearing TR chunks...")
        DocumentChunk.objects.filter(regime='tr').delete()
        DocumentSection.objects.all().delete()
    else:
        print("🧹 Clearing EU chunks...")
        DocumentChunk.objects.filter(regime='eu').delete()

    docs_to_ingest: list[tuple[str, dict]] = []
    if regime in ('tr', 'all'):
        docs_to_ingest += [('tr', d) for d in TR_DOCUMENTS]
    if regime in ('eu', 'all'):
        docs_to_ingest += [('eu', d) for d in EU_DOCUMENTS]

    all_chunks: list[DocumentChunk] = []
    global_idx = 1

    for regime_tag, doc in docs_to_ingest:
        file_path = os.path.join(settings.BASE_DIR, doc["file"])
        if regime_tag == 'tr':
            chunks, global_idx = _parse_single_document(file_path, doc["name"], doc["short"], global_idx)
        else:
            chunks, global_idx = _parse_eu_document(file_path, doc["name"], doc["short"], global_idx)
        for c in chunks:
            c.regime = regime_tag
        all_chunks.extend(chunks)

    print(f"\n🧠 Generating embeddings for {len(all_chunks)} chunks with multilingual-e5-large...")
    from sentence_transformers import SentenceTransformer as _ST
    _ingest_model = _ST("intfloat/multilingual-e5-large")

    content_for_embedding = [f"passage: {c.content}" for c in all_chunks]
    vectors = _ingest_model.encode(
        content_for_embedding, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    for chunk, vector in zip(all_chunks, vectors):
        chunk.embedding = vector.tolist()

    print("💾 Saving to database...")
    DocumentChunk.objects.bulk_create(all_chunks)
    total_docs = len(docs_to_ingest)
    print(f"✅ Done — {len(all_chunks)} chunks saved across {total_docs} documents.")


# ── Query pipeline ───────────────────────────────────────────────────────────

def _log_tokens(label: str, response) -> dict:
    usage = response.usage_metadata or {}
    inp   = usage.get("input_tokens", 0)
    out   = usage.get("output_tokens", 0)
    total = usage.get("total_tokens", inp + out)
    return {"input": inp, "output": out, "total": total}


MADDE_REGEX   = re.compile(r'\bMadde\s+(\d{1,3})\b', re.IGNORECASE)
BOLUM_REGEX   = re.compile(r'\b(\d{1,2})\s*\.?\s*[Bb]ölüm\b|\b[Bb]ölüm\s+(\d{1,2})\b')
ARTICLE_REGEX = re.compile(r'\bArticle\s+(\d{1,3})\b', re.IGNORECASE)


def _extract_filters_by_regex(text: str) -> dict:
    madde_match = MADDE_REGEX.search(text)
    bolum_match = BOLUM_REGEX.search(text)
    madde = int(madde_match.group(1)) if madde_match else None
    bolum = int(bolum_match.group(1) or bolum_match.group(2)) if bolum_match else None
    return {"madde": madde, "bolum": bolum}


def _extract_eu_filters(text: str) -> dict:
    article_match = ARTICLE_REGEX.search(text)
    return {"madde": int(article_match.group(1)) if article_match else None, "bolum": None}


@traceable(name="analyze_query")
def analyze_query(question: str, history: list) -> dict:
    filters = _extract_filters_by_regex(question)
    return {"madde": filters["madde"], "bolum": filters["bolum"]}


@traceable(name="retrieval_pipeline")
def retrieval_pipeline_traced(question: str, vector_chunks: list) -> dict:
    return {
        "question": question,
        "results": [
            {
                "rank": i + 1,
                "document": c.document_name,
                "madde": c.madde,
                "bolum": c.section.number if c.section else None,
                "madde_title": c.madde_title,
                "preview": c.content[:120],
            }
            for i, c in enumerate(vector_chunks)
        ],
        "stats": {"count": len(vector_chunks)},
    }


_TR_SYSTEM_PROMPT = (
    "You are an expert on Turkish personal data protection law. "
    "Your knowledge covers the following documents:\n"
    "1. KVKK — 6698 sayılı Kişisel Verilerin Korunması Kanunu (main law)\n"
    "2. Aktarım Yönetmeliği — Kişisel Verilerin Yurt Dışına Aktarılmasına İlişkin Usul ve Esaslar Hakkında Yönetmelik\n"
    "3. Silme Yönetmeliği — Kişisel Verilerin Silinmesi, Yok Edilmesi veya Anonim Hale Getirilmesi Hakkında Yönetmelik\n"
    "4. Aydınlatma Tebliği — Aydınlatma Yükümlülüğünün Yerine Getirilmesinde Uyulacak Usul ve Esaslar Hakkında Tebliğ\n"
    "5. VERBİS Yönetmeliği — Veri Sorumluları Sicili Hakkında Yönetmelik\n\n"
    "Answer strictly based on the provided context passages. "
    "If the question is not related to Turkish personal data protection law or the documents listed above, "
    "respond only with: 'Bu soru KVKK mevzuatı kapsamında değildir.' — do not answer it. "
    "If the answer is not found in the context, say so — do not speculate. "
    "Always cite the document name and Madde number (e.g. 'KVKK Madde 6', 'Aktarım Yönetmeliği Madde 9'). "
    "Reply in the same language the user used. "
    "Use the conversation history for follow-up questions.\n\n"
    "Examples of ideal citations:\n"
    "- 'KVKK Madde 5'e göre...'\n"
    "- 'Aktarım Yönetmeliği Madde 3 uyarınca...'\n"
    "- 'Aydınlatma Tebliği Madde 4 kapsamında...'"
)

_EU_SYSTEM_PROMPT = (
    "You are an expert on EU data protection and AI regulation law. "
    "Your knowledge covers the following documents:\n"
    "1. GDPR — General Data Protection Regulation (EU) 2016/679\n"
    "2. ePrivacy Directive — Directive 2002/58/EC concerning the processing of personal data and the protection of privacy in the electronic communications sector\n"
    "3. EU AI Act — Regulation (EU) 2024/1689 on artificial intelligence\n\n"
    "Answer strictly based on the provided context passages. "
    "If the question is not related to EU data protection or AI regulation, "
    "respond only with: 'This question is not within the scope of EU data protection or AI regulation.' — do not answer it. "
    "If the answer is not found in the context, say so — do not speculate. "
    "Always cite the document name and Article number (e.g. 'GDPR Article 6', 'EU AI Act Article 10'). "
    "Reply in the same language the user used. "
    "Use the conversation history for follow-up questions.\n\n"
    "Examples of ideal citations:\n"
    "- 'Under GDPR Article 6...'\n"
    "- 'According to EU AI Act Article 10...'\n"
    "- 'As defined in GDPR Article 4...'"
)


@traceable(name="query_kvkk")
def query_kvkk(question: str, history: list | None = None, regime: str = 'tr') -> dict:
    if history is None:
        history = []
    history = history[-6:]

    # 1. Extract article/madde filters from question text
    if regime == 'eu':
        filters = _extract_eu_filters(question)
    else:
        analysis = analyze_query(question, history)
        filters  = {"madde": analysis["madde"], "bolum": analysis["bolum"]}

    # 2. Embed question — E5 requires "query: " prefix at query time
    question_embedding = embed_query("query: " + question)

    # 3. Build queryset filtered by regime; exclude Mülga (repealed) paragraphs
    queryset = (
        DocumentChunk.objects
        .select_related('section')
        .filter(regime=regime)
        .exclude(content__icontains='Mülga')
    )
    if filters["madde"] is not None:
        queryset = queryset.filter(madde=filters["madde"])
    if filters.get("bolum") is not None:
        queryset = queryset.filter(section__number=filters["bolum"])

    # 4. Retrieve: specific article → all its chunks in order; general → vector top-10
    if filters["madde"] is not None:
        similar_chunks = list(queryset.order_by('chunk_index'))
    else:
        similar_chunks = list(
            queryset
            .annotate(distance=CosineDistance('embedding', question_embedding))
            .order_by('distance')[:10]
        )
        retrieval_pipeline_traced(question, similar_chunks)

    # 5. Out-of-scope check
    oos_answer = (
        "Bu soru KVKK mevzuatı kapsamında değil veya belgede bu konuda bilgi bulunmuyor."
        if regime == 'tr'
        else "This question is not within the scope of EU data protection or AI regulation, or the information is not available in the provided documents."
    )

    if not similar_chunks:
        return {"answer": oos_answer, "sources": []}
    if filters["madde"] is None:
        if getattr(similar_chunks[0], 'distance', 0) > 0.6:
            return {"answer": oos_answer, "sources": []}

    # 6. Build context
    context = "\n\n".join(c.content for c in similar_chunks)

    # 7. Call Gemini
    llm = ChatGoogleGenerativeAI(
        model=ANSWER_MODEL,
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

    system_prompt = _EU_SYSTEM_PROMPT if regime == 'eu' else _TR_SYSTEM_PROMPT

    messages = [
        SystemMessage(content=system_prompt),
        *history_messages,
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    _log_tokens("answer", response)

    # Deduplicate sources by (document, madde) and sort in document order
    seen = set()
    ordered_sources = []
    for c in sorted(
        similar_chunks,
        key=lambda x: (x.document_name, x.section.number if x.section else 99, x.madde or 99)
    ):
        key = (c.document_name, c.madde)
        if key not in seen:
            seen.add(key)
            ordered_sources.append(c)

    return {
        "answer": response.content,
        "model":  ANSWER_MODEL,
        "regime": regime,
        "sources": [
            {
                "document_name": c.document_name,
                "bolum":         c.section.number if c.section else None,
                "madde":         c.madde,
                "madde_title":   c.madde_title,
                "content":       c.content.split('\n', 1)[1].strip() if '\n' in c.content else c.content,
            }
            for c in ordered_sources
        ],
    }
