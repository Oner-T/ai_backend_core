import os
import re
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from intelligence.models import DocumentChunk, DocumentSection

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
    current_madde_text = "Başlangıç Hükümleri"
    
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
                # Create the metadata record in the database
                current_section_obj, created = DocumentSection.objects.get_or_create(name=bolum_name)
                print(f"📍 Found Section: {bolum_name}")
                continue # Skip adding the title itself as a standalone chunk

            # 2. Check if the line is a new MADDE
            madde_match = madde_pattern.match(clean_line)
            if madde_match:
                current_madde_text = madde_match.group(1)
                print(f"   ↳ Found Article: {current_madde_text}")

            # 3. Build the paragraph
            current_paragraph.append(clean_line)

            # If the line ends with a period or semicolon, assume the thought is complete
            if clean_line.endswith('.') or clean_line.endswith(';'):
                full_text = " ".join(current_paragraph)
                
                chunk = DocumentChunk(
                    section=current_section_obj, # Linking to the metadata table
                    madde=current_madde_text,    # Storing the specific article
                    document_name="KVKK.pdf",
                    chunk_index=global_chunk_index,
                    content=full_text
                )
                chunk_objects.append(chunk)
                
                global_chunk_index += 1
                current_paragraph = [] # Reset for the next paragraph

    print(f"🧠 Generating {len(chunk_objects)} Vector Embeddings locally via HuggingFace...")
    # Initialize the free, local embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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