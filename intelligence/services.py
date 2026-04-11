import os
import re
import json
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pgvector.django import CosineDistance
from intelligence.models import DocumentChunk, DocumentSection

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

    # Find the PDF in the root directory next to manage.py
    file_path = os.path.join(settings.BASE_DIR, 'KVKK.pdf')

    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        print("Please make sure KVKK.pdf is in the same folder as manage.py")
        return

    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Regex Patterns for KVKK structure
    bolum_pattern = re.compile(r"^([A-ZÇĞİÖŞÜ]+\sBÖLÜM)") 
    madde_pattern = re.compile(r"^(MADDE\s\d+)")

    # "State" trackers
    current_section_obj = None
    current_madde_num = None
    
    chunk_objects = []
    global_chunk_index = 1

    print("✂️ Parsing structure and creating chunks...")
    for page in pages:
        # Split page by newlines to process line-by-line
        lines = page.page_content.split('\n')
        
        current_paragraph = []

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # 1. Check if the line is a new BÖLÜM
            bolum_match = bolum_pattern.match(clean_line)
            if bolum_match:
                bolum_name = bolum_match.group(1)
                bolum_number = bolum_name_to_int(bolum_name)
                # Create the metadata record in the database
                current_section_obj, created = DocumentSection.objects.get_or_create(number=bolum_number)
                print(f"📍 Found Section: {bolum_name} → {bolum_number}")
                continue # Skip adding the title itself as a standalone chunk

            # 2. Check if the line is a new MADDE
            madde_match = madde_pattern.match(clean_line)
            if madde_match:
                current_madde_num = int(re.search(r'\d{1,3}', madde_match.group(1)).group())
                print(f"   ↳ Found Article: MADDE {current_madde_num}")

            # 3. Build the paragraph
            current_paragraph.append(clean_line)

            # If the line ends with a period or semicolon, assume the thought is complete
            if clean_line.endswith('.') or clean_line.endswith(';'):
                full_text = " ".join(current_paragraph)
                
                chunk = DocumentChunk(
                    section=current_section_obj, # Linking to the metadata table
                    madde=current_madde_num,
                    document_name="KVKK.pdf",
                    chunk_index=global_chunk_index,
                    content=full_text
                )
                chunk_objects.append(chunk)
                
                global_chunk_index += 1
                current_paragraph = [] # Reset for the next paragraph

    print(f"🧠 Generating {len(chunk_objects)} Vector Embeddings locally via HuggingFace...")
    # Initialize the free, local embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Extract just the text from our chunks to send to the model
    texts_to_embed = [chunk.content for chunk in chunk_objects]
    
    # Generate all vectors using your CPU
    vector_lists = embeddings_model.embed_documents(texts_to_embed)
    
    # Attach the generated vectors back to our Django objects
    for chunk, vector in zip(chunk_objects, vector_lists):
        chunk.embedding = vector

    print("💾 Saving chunks AND vectors to Supabase...")
    # Hit the database exactly once to save all chunks
    DocumentChunk.objects.bulk_create(chunk_objects)
    
    print(f"✅ Successfully saved {len(chunk_objects)} highly intelligent chunks to Supabase!")


def parse_query_filters(question: str) -> dict:
    """Extract explicit madde/bolum references from the question using the LLM."""
    llm = ChatOllama(model="llama3.2", temperature=0, format="json")
    messages = [
        SystemMessage(content=(
            "You are a metadata extractor for KVKK (Turkish Personal Data Protection Law) queries. "
            "Analyze the user's question and extract any explicit references to a specific article (madde) "
            "or chapter (bölüm) number. "
            "Return a JSON object with exactly two keys: 'madde' and 'bolum'. "
            "The value should be an integer if explicitly mentioned, or null if not. "
            "Examples:\n"
            "  'Madde 5 nedir?' → {\"madde\": 5, \"bolum\": null}\n"
            "  'İkinci bölüm hakkında bilgi ver' → {\"madde\": null, \"bolum\": 2}\n"
            "  'Kişisel veri nedir?' → {\"madde\": null, \"bolum\": null}\n"
            "  'MADDE 12 ve bölüm 3' → {\"madde\": 12, \"bolum\": 3}\n"
            "Return only valid JSON, nothing else."
        )),
        HumanMessage(content=question),
    ]
    try:
        response = llm.invoke(messages)
        filters = json.loads(response.content)
        madde = filters.get("madde")
        bolum = filters.get("bolum")
        # Validate they are integers or None
        madde = int(madde) if madde is not None else None
        bolum = int(bolum) if bolum is not None else None
        print(f"🔍 Parsed filters — madde: {madde}, bolum: {bolum}")
        return {"madde": madde, "bolum": bolum}
    except Exception:
        return {"madde": None, "bolum": None}


def rewrite_query(question: str) -> str:
    llm = ChatOllama(model="llama3.2", temperature=0)
    messages = [
        SystemMessage(content=(
            "You are a search query optimizer for KVKK (Turkish Personal Data Protection Law) document retrieval. "
            "Rewrite the user's question into a richer, more specific search query that will retrieve the most relevant "
            "passages from a vector database containing KVKK legal text. "
            "Expand abbreviations, add relevant legal keywords, and clarify vague terms. "
            "Keep the output language the same as the input (Turkish in → Turkish out, English in → English out). "
            "Return only the rewritten query, nothing else."
        )),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    rewritten = response.content.strip()
    print(f"🔄 Query rewritten: '{question}' → '{rewritten}'")
    return rewritten


def query_kvkk(question: str) -> dict:
    # 1. Parse explicit madde/bolum references and rewrite query — run logic sequentially
    filters = parse_query_filters(question)
    rewritten_question = rewrite_query(question)

    # 2. Embed the rewritten query
    embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    question_embedding = embeddings_model.embed_query(rewritten_question)

    # 3. Build queryset — apply metadata filters if the user referenced a specific madde/bolum
    queryset = DocumentChunk.objects.select_related('section')
    if filters["madde"] is not None:
        queryset = queryset.filter(madde=filters["madde"])
        print(f"📌 Filtering by madde={filters['madde']}")
    if filters["bolum"] is not None:
        queryset = queryset.filter(section__number=filters["bolum"])
        print(f"📌 Filtering by bolum={filters['bolum']}")

    # 4. Retrieve strategy:
    #    - Specific madde referenced → fetch ALL chunks for that madde in document order
    #      (guarantees completeness, no missing sub-items)
    #    - General question → top-5 cosine similarity search
    if filters["madde"] is not None:
        similar_chunks = list(queryset.order_by('chunk_index'))
        print(f"📄 Retrieved all {len(similar_chunks)} chunks for madde={filters['madde']}")
    else:
        similar_chunks = list(
            queryset
            .annotate(distance=CosineDistance('embedding', question_embedding))
            .order_by('distance')[:5]
        )

    # 5. If best match is too far away, the question is out of scope
    #    (skip threshold check when madde filter was applied — we trust the metadata)
    first_distance = getattr(similar_chunks[0], 'distance', 0) if similar_chunks else 1
    if not similar_chunks or (filters["madde"] is None and first_distance > 0.6):
        return {
            "answer": "Bu soru KVKK kapsamında değil veya belgede bu konuda bilgi bulunmuyor.",
            "sources": []
        }

    # 6. Build context from retrieved chunks with metadata
    context_parts = []
    for chunk in similar_chunks:
        bolum = chunk.section.number if chunk.section else "?"
        madde = chunk.madde or "?"
        context_parts.append(f"[Bölüm {bolum}, Madde {madde}]\n{chunk.content}")
    context = "\n\n".join(context_parts)

    # 5. Call local Ollama LLM
    llm = ChatOllama(model="llama3.2", temperature=0)

    messages = [
        SystemMessage(content=(
            "You are an expert on KVKK (Kişisel Verilerin Korunması Kanunu — Turkish Personal Data Protection Law). "
            "Answer the user's question strictly based on the provided KVKK context passages. "
            "If the answer is not present in the context, say so clearly — do not speculate or use outside knowledge. "
            "Always cite the Bölüm (Chapter) and Madde (Article) your answer is based on. "
            "Reply in the same language the user used (Turkish question → Turkish answer, English question → English answer)."
            "When user asks for all information about a specific topic, give the all relevant passages from the context that mention that topic, along with their citations."
        )),
        HumanMessage(content=f"KVKK Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [
            {"bolum": c.section.number if c.section else None, "madde": c.madde}
            for c in similar_chunks
        ],
    }