from pathlib import Path
from typing import List, Dict, Tuple

from PyPDF2 import PdfReader
import PyPDF2  

def load_pdf_text(path: Path) -> List[Dict]:
    """
    Return a list of {"page_number": int, "text": str}
    """
    pages: List[Dict] = []
    # TODO: open the PDF with PyPDF2 and extract text page by page
    # Hints:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    for i in range(page_count):
        page = reader.pages[i]
        text = page.extract_text() or  ""
        pages.append({"page_number": i + 1, "text": text})
    return pages

def chunk_pages(
    pages: List[Dict],
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Dict]:
    """
    Turn page-level text into overlapping chunks that respect page boundaries.
    Each chunk: {"id": str, "page_range": (start_page, end_page), "text": str}
    """
    chunks: List[Dict] = []

    current_text_parts: List[str] = []
    current_start_page: int | None = None
    current_end_page: int | None = None

    def flush_chunk():
        nonlocal current_text_parts, current_start_page, current_end_page
        if not current_text_parts or current_start_page is None:
            return
        full_text = "".join(current_text_parts).strip()
        if not full_text:
            return

        chunk_id = f"pages-{current_start_page}-{current_end_page}"
        chunks.append({
            "id": chunk_id,
            "page_range": (current_start_page, current_end_page),
            "text": full_text,
        })

        # prepare overlap for next chunk
        if len(full_text) > overlap_chars:
            overlap_text = full_text[-overlap_chars:]
        else:
            overlap_text = full_text

        current_text_parts = [overlap_text]
        current_start_page = current_end_page  
        # start_page for the next chunk stays the same or moves forward?
        # TODO: decide: should next chunk's start_page be current_start_page or current_end_page?
        # For a legal doc, it's usually okay to use current_start_page for context, or just current_end_page.

    for page in pages:
        page_num = page["page_number"]
        text = (page["text"] or "").strip()
        if not text:
            continue

        if current_start_page is None:
            current_start_page = page_num

        # Split page into paragraphs or lines to avoid mid-sentence splits
        segments = text.split("\n\n")  # naive paragraph split

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # If adding this seg would exceed max_chars, flush current chunk first
            candidate = "".join(current_text_parts) + ("\n\n" if current_text_parts else "") + seg
            if len(candidate) > max_chars:
                current_end_page = page_num
                flush_chunk()
                # After flush, start fresh (we already put overlap into current_text_parts)
                if current_start_page is None:
                    current_start_page = page_num

            # Now safely add this segment
            if current_text_parts:
                current_text_parts.append("\n\n" + seg)
            else:
                current_text_parts.append(seg)

            current_end_page = page_num

    # flush remaining text
    current_end_page = current_end_page or (pages[-1]["page_number"] if pages else 0)
    flush_chunk()

    return chunks

if __name__ == "__main__":
    pdf_path = Path("Topic2.1.pdf")  # replace with your real file
    pages = load_pdf_text(pdf_path)
    print(f"Loaded {len(pages)} pages")

    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks")
    for c in chunks[:3]:
        print(c["id"], len(c["text"]))
    print("--- CHUNK 0 ---")
    print(chunks[0]["text"][:300])
    print("--- CHUNK 1 ---")
    print(chunks[1]["text"][:300])


    
