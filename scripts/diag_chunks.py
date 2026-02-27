"""Diagnostic: print all Doc02 chunks showing section_header and text start."""

from pathlib import Path

from src.pipelines.chunker import chunk_document_docling
from src.pipelines.models import DocumentChunk

doc_path = Path("data/raw/Doc02_Radiation_Sickness_Symptom_Guide.docx")
meta = DocumentChunk(
    chunk_id="placeholder",
    text="",
    source_document="Doc02",
    section_header="",
    department="General",
    access_level="General Employee",
    effective_date=None,
    document_status="ACTIVE",
    content_type="narrative",
    chunk_index=0,
    total_chunks=1,
    token_count=0,
)
chunks = chunk_document_docling(doc_path, 512, "doc02", meta)
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    pipe_ok = "|" in c.text if c.content_type == "table" else "n/a"
    kv_bad = "** =" in c.text
    print(
        f"chunk_{c.chunk_index:02d} | {c.content_type:12s} | "
        f"header={repr(c.section_header[:40]):45s} | "
        f"pipe={pipe_ok} | kv_bad={kv_bad} | "
        f"text={repr(c.text[:60])}"
    )
