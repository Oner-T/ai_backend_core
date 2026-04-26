import os
import re
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pgvector.django import CosineDistance
from langsmith import traceable
from intelligence.models import DocumentChunk, DocumentSection

_embeddings_model: HuggingFaceEmbeddings | None = None

ANSWER_MODEL = "gemini-2.5-flash"

def get_embeddings_model() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        # 2.24GB model — load once per process, reuse for all requests
        _embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return _embeddings_model


# ── Document registry ────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "file":  "KVKK.pdf",
        "name":  "KVKK",
        "short": "KVKK",
    },
    {
        "file":  "KVKK_son.pdf",
        "name":  "Yurt Dışına Aktarım Yönetmeliği",
        "short": "Aktarım Yönetmeliği",
    },
    {
        "file":  "SilmeYonetmelik_KVKK.pdf",
        "name":  "Silme Yönetmeliği",
        "short": "Silme Yönetmeliği",
    },
    {
        "file":  "AydınlatmaYükümlülüğününYerineGetirilmesindeUyulacakUsulveEsaslarHakkındaTebliğ_KVKK.pdf",
        "name":  "Aydınlatma Tebliği",
        "short": "Aydınlatma Tebliği",
    },
    {
        "file":  "VeriSorumlularıSiciliHakkındaYönetmelik_KVKK.pdf",
        "name":  "VERBİS Yönetmeliği",
        "short": "VERBİS Yönetmeliği",
    },
]

# Some PDFs (Aydınlatma Tebliği) encode digits as Unicode private-use characters
# via a custom font. Map them back to ASCII digits before any pattern matching.
_PRIVATE_DIGIT_MAP = str.maketrans({
    '\uf7a0': '0', '\uf7a1': '1', '\uf7a2': '2', '\uf7a3': '3',
    '\uf7a4': '4', '\uf7a5': '5', '\uf7a6': '6', '\uf7a7': '7',
    '\uf7a8': '8', '\uf7a9': '9',
    '\xa0': ' ',   # non-breaking space → regular space
})

def _normalize(line: str) -> str:
    return line.translate(_PRIVATE_DIGIT_MAP)


# Lines to skip that appear in web-scraped PDFs (navigation, timestamps, URLs)
_WEB_NOISE = re.compile(
    r'^(TR|EN|DE)$'
    r'|^\s*/'                          # breadcrumbs: " /Yönetmelikler/"
    r'|^Yayınlanma Tarihi'             # publication date header
    r'|^M\s*ENÜ\s*$'                  # menu button ("M ENÜ" or "MENÜ")
    r'|^\d{1,2}/\d{2}/\d{2},\s*\d'   # timestamps: "4/22/26, 8:15 AM"
    r'|https?://'                      # URLs
)

# When these appear, we've hit the website footer — stop parsing entirely
_FOOTER_STOP = re.compile(
    r'〉'                                # navigation arrows
    r'|^Yönetmelikte Değişiklik Yapan'  # amendment history appendix
    r'|KİŞİSEL VERİLERİ KORUMA KURUMU' # institution name in footer
)

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

def bolum_name_to_int(bolum_text: str) -> int | None:
    ordinal = bolum_text.replace("BÖLÜM", "").strip()
    return TURKISH_ORDINAL_TO_INT.get(ordinal)


# ── Document parser ──────────────────────────────────────────────────────────

def _parse_single_document(
    file_path: str,
    document_name: str,
    doc_short_name: str,
    global_chunk_index: int,
) -> tuple[list, int]:
    """Parse one legal PDF into DocumentChunk objects without saving to DB.

    Returns (chunk_objects, next_global_chunk_index).
    Handles standard MADDE/BÖLÜM structure, web-scraped boilerplate,
    and PDFs where MADDE numbers are missing in the extracted text.
    """
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return [], global_chunk_index

    print(f"\n📄 Loading: {file_path}")
    pages = PyPDFLoader(file_path).load()

    # Patterns
    # BÖLÜM: normal "BİRİNCİ BÖLÜM" or standalone "BÖLÜM" (split across lines in some PDFs)
    bolum_pattern    = re.compile(r"^([A-ZÇĞİÖŞÜ]+\s+BÖLÜM)$|^(BÖLÜM)$")
    madde_pattern    = re.compile(r"^MADDE\s+(\d{1,3})")          # with number
    madde_no_num     = re.compile(r"^MADDE\s*[\-–]")              # number missing (some PDFs)
    num_para_pattern = re.compile(r"^\((\d+)\)")
    footnote_pattern = re.compile(r"^\d+\s+\d{1,2}/\d{1,2}/\d{4}")
    def_item_pattern = re.compile(r"^([a-zçğışöü])\)\s+\S")

    # State
    current_section_obj  = None
    current_madde_num    = None
    current_madde_title  = None
    last_seen_line       = None
    paragraph_buffer: list[str] = []
    chunk_objects: list[DocumentChunk] = []
    madde_seq            = 0   # sequential counter for PDFs with missing madde numbers
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

        # Stop if we've reached website footer content
        if _FOOTER_STOP.search(clean_line):
            flush_chunk()
            stop_parsing = True
            break

        # Skip web navigation noise and footnote lines
        if _WEB_NOISE.match(clean_line):
            continue
        if footnote_pattern.match(clean_line):
            continue

        # ── New BÖLÜM ──────────────────────────────────────────────────────
        bolum_match = bolum_pattern.match(clean_line)
        if bolum_match:
            flush_chunk()
            current_madde_num   = None
            current_madde_title = None

            if bolum_match.group(1):
                # "BİRİNCİ BÖLÜM" on one line
                bolum_name = bolum_match.group(1)
            else:
                # Standalone "BÖLÜM" — ordinal was on the previous line
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

        # ── New MADDE ───────────────────────────────────────────────────────
        madde_match   = madde_pattern.match(clean_line)
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
                # Number missing in this PDF — use sequential counter
                madde_seq += 1
                current_madde_num = madde_seq

            current_madde_title = candidate_title
            last_seen_line      = clean_line
            paragraph_buffer.append(clean_line)
            print(f"      ↳ MADDE {current_madde_num} — {current_madde_title}")
            continue

        # ── Numbered paragraph (2), (3)… → chunk boundary ──────────────────
        num_match = num_para_pattern.match(clean_line)
        if num_match and current_madde_num is not None:
            para_num = int(num_match.group(1))
            if para_num > 1:
                flush_chunk()
                paragraph_buffer.append(clean_line)
                last_seen_line = clean_line
                continue

        # ── Tanımlar madde: lettered items a), b), c)… → each is its own chunk
        is_tanim = (
            current_madde_title is not None
            and 'anım' in current_madde_title  # matches "Tanımlar", "Tanım"
        )
        if is_tanim and def_item_pattern.match(clean_line):
            flush_chunk()
            paragraph_buffer.append(clean_line)
            last_seen_line = clean_line
            continue

        # ── Regular content line ────────────────────────────────────────────
        if current_madde_num is not None:
            paragraph_buffer.append(clean_line)

        last_seen_line = clean_line

    flush_chunk()
    print(f"  ✅ {len(chunk_objects)} chunks parsed from {document_name}")
    return chunk_objects, global_chunk_index


def ingest_all_documents():
    """Clear DB and re-ingest all 5 KVKK-related documents."""
    print("🧹 Clearing existing data...")
    DocumentChunk.objects.all().delete()
    DocumentSection.objects.all().delete()

    all_chunks: list[DocumentChunk] = []
    global_idx = 1

    for doc in DOCUMENTS:
        file_path = os.path.join(settings.BASE_DIR, doc["file"])
        chunks, global_idx = _parse_single_document(
            file_path, doc["name"], doc["short"], global_idx
        )
        all_chunks.extend(chunks)

    print(f"\n🧠 Generating embeddings for {len(all_chunks)} total chunks with multilingual-e5-large...")
    from sentence_transformers import SentenceTransformer as _ST
    _ingest_model = _ST("intfloat/multilingual-e5-large")

    # e5-large + WITH HEADERS = MRR 0.904 (ablation-tested)
    content_for_embedding = [f"passage: {c.content}" for c in all_chunks]
    vectors = _ingest_model.encode(
        content_for_embedding, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    for chunk, vector in zip(all_chunks, vectors):
        chunk.embedding = vector.tolist()

    print("💾 Saving to database...")
    DocumentChunk.objects.bulk_create(all_chunks)
    print(f"✅ Done — {len(all_chunks)} chunks saved across {len(DOCUMENTS)} documents.")


# ── Query pipeline ───────────────────────────────────────────────────────────

def _log_tokens(label: str, response) -> dict:
    usage = response.usage_metadata or {}
    inp   = usage.get("input_tokens", 0)
    out   = usage.get("output_tokens", 0)
    total = usage.get("total_tokens", inp + out)
    return {"input": inp, "output": out, "total": total}


MADDE_REGEX = re.compile(r'\bMadde\s+(\d{1,3})\b', re.IGNORECASE)
BOLUM_REGEX = re.compile(r'\b(\d{1,2})\s*\.?\s*[Bb]ölüm\b|\b[Bb]ölüm\s+(\d{1,2})\b')


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


@traceable(name="retrieval_pipeline")
def retrieval_pipeline_traced(question: str, vector_chunks: list) -> dict:
    """LangSmith-visible summary of retrieval results."""
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


@traceable(name="query_kvkk")
def query_kvkk(question: str, history: list | None = None) -> dict:
    if history is None:
        history = []
    history = history[-6:]

    # 1. Extract madde/bolum filters from question text
    analysis = analyze_query(question, history)
    filters = {"madde": analysis["madde"], "bolum": analysis["bolum"]}

    # 2. Embed question — E5 requires "query: " prefix at query time
    question_embedding = get_embeddings_model().embed_query("query: " + question)

    # 3. Build queryset — exclude Mülga (repealed) paragraphs
    queryset = DocumentChunk.objects.select_related('section').exclude(content__icontains='Mülga')
    if filters["madde"] is not None:
        queryset = queryset.filter(madde=filters["madde"])
    if filters["bolum"] is not None:
        queryset = queryset.filter(section__number=filters["bolum"])

    # 4. Retrieve: specific madde → all its chunks in order; general → vector top-10
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
    if not similar_chunks:
        return {
            "answer": "Bu soru KVKK mevzuatı kapsamında değil veya belgede bu konuda bilgi bulunmuyor.",
            "sources": []
        }
    if filters["madde"] is None:
        if getattr(similar_chunks[0], 'distance', 0) > 0.6:
            return {
                "answer": "Bu soru KVKK mevzuatı kapsamında değil veya belgede bu konuda bilgi bulunmuyor.",
                "sources": []
            }

    # 6. Build context — use the stored chunk content directly (header is already embedded)
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

    messages = [
        SystemMessage(content=(
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
        )),
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
        "model": ANSWER_MODEL,
        "sources": [
            {
                "document_name": c.document_name,
                "bolum": c.section.number if c.section else None,
                "madde": c.madde,
                "madde_title": c.madde_title,
                # Strip the header line prepended during ingestion
                "content": c.content.split('\n', 1)[1].strip() if '\n' in c.content else c.content,
            }
            for c in ordered_sources
        ],
    }
