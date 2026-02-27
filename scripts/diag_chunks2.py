"""Debug: what doc_item types and labels do chunks 3 and 7 have in Doc02?"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

doc_path = Path("data/raw/Doc02_Radiation_Sickness_Symptom_Guide.docx")
converter = DocumentConverter()
result = converter.convert(str(doc_path))
docling_doc = result.document

chunker = HybridChunker(
    tokenizer="BAAI/bge-small-en-v1.5",
    max_tokens=512,
    merge_peers=True,
)
raw_chunks = list(chunker.chunk(docling_doc))

# Print tables in document
print(f"docling_doc.tables count: {len(docling_doc.tables)}")
for i, tbl in enumerate(docling_doc.tables):
    print(f"  table[{i}] self_ref={getattr(tbl, 'self_ref', 'N/A')}")

# Inspect chunks 3 and 7
for idx in [3, 7]:
    chunk = raw_chunks[idx]
    print(f"\n--- chunk {idx} ---")
    print(f"  text[:80]: {repr(chunk.text[:80])}")
    try:
        for item in chunk.meta.doc_items:
            print(
                f"  doc_item type={type(item).__name__}  label={getattr(item, 'label', 'N/A')}  self_ref={getattr(item, 'self_ref', 'N/A')}"
            )
    except Exception as e:
        print(f"  error reading doc_items: {e}")
