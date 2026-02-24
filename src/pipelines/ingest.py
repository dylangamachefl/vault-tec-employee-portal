"""
Vault-Tec Employee Portal — Document Ingestion Pipeline (Docling)
=================================================================

Stage 1:  Docling DocumentConverter  (raw files → DoclingDocument)
Stage 2:  Docling HybridChunker      (DoclingDocument → list[DocChunk])
Stage 3:  Metadata enrichment        (DocChunk → DocumentChunk)
Stage 4:  Persist to ChromaDB + JSON backup

CLI usage:
    uv run python -m src.pipelines.ingest --chunk-size 512
    uv run python -m src.pipelines.ingest --chunk-size 1024 --collection vault_documents_1024
    uv run python -m src.pipelines.ingest --experiment          # runs 512 + 1024
    uv run python -m src.pipelines.ingest --chunk-size 512 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from src.pipelines.chunker import chunk_document_docling
from src.pipelines.models import DocumentChunk
from src.pipelines.persist import persist_to_chroma, write_json_backup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Document metadata registry
# Maps doc_slug (filename stem, lowercase, underscores) → metadata fields.
# Slugs must match Path(filename).stem.lower().replace(" ", "_").replace("-", "_")
# ---------------------------------------------------------------------------

DOCUMENT_METADATA: dict[str, dict] = {
    "doc01_vault_dweller_code_of_conduct": {
        "department": "General",
        "access_level": "General Employee",
        "effective_date": date(2076, 5, 1),
        "document_status": "ACTIVE",
    },
    "doc02_radiation_sickness_symptom_guide": {
        "department": "General",
        "access_level": "General Employee",
        "effective_date": date(2076, 8, 1),
        "document_status": "ACTIVE",
    },
    "doc3_emergency_evacuation_procedures": {
        "department": "General",
        "access_level": "General Employee",
        "effective_date": date(2076, 9, 1),
        "document_status": "ACTIVE",
    },
    "doc4_overseer_compensation_hr": {
        "department": "HR",
        "access_level": "HR Restricted",
        "effective_date": date(2076, 1, 1),
        "document_status": "ACTIVE",
    },
    "vaulttec_hr_doc5_nvdr": {
        "department": "HR",
        "access_level": "HR Restricted",
        "effective_date": date(2076, 3, 1),
        "document_status": "ACTIVE",
    },
    "doc06_goat_exam_administration": {
        "department": "HR",
        "access_level": "HR Restricted",
        "effective_date": date(2076, 6, 1),
        "document_status": "ACTIVE",
    },
    "doc7_vault76_tricentennial_promotional_strategy": {
        "department": "Marketing",
        "access_level": "Marketing Eyes Only",
        "effective_date": date(2076, 2, 1),
        "document_status": "ACTIVE",
    },
    "doc08_geck_advertising_guidelines": {
        "department": "Marketing",
        "access_level": "Marketing Eyes Only",
        "effective_date": date(2076, 4, 1),
        "document_status": "ACTIVE",
    },
    "doc9_endofworld_crisis_response_messaging": {
        "department": "Marketing",
        "access_level": "Marketing Eyes Only",
        "effective_date": date(2077, 10, 1),
        "document_status": "ACTIVE",
    },
    "doc10_vault_door_override_protocol": {
        "department": "Admin",
        "access_level": "Admin Eyes Only",
        "effective_date": date(2076, 12, 1),
        "document_status": "ACTIVE",
    },
    "doc11_zax_mainframe_root_access": {
        "department": "Admin",
        "access_level": "Admin Eyes Only",
        "effective_date": date(2076, 7, 1),
        "document_status": "ACTIVE",
    },
    "doc12_surface_exploration_archived": {
        "department": "General",
        "access_level": "General Employee",
        "effective_date": date(2077, 1, 1),
        "document_status": "ARCHIVED",
    },
    "doc13_surface_exploration_active": {
        "department": "General",
        "access_level": "General Employee",
        "effective_date": date(2077, 11, 1),
        "document_status": "ACTIVE",
    },
    "sop_pipboy_2000_calibration": {
        "department": "Admin",
        "access_level": "Admin Eyes Only",
        "effective_date": date(2075, 3, 1),
        "document_status": "ACTIVE",
    },
    "sop_pipboy_3000_calibration": {
        "department": "Admin",
        "access_level": "Admin Eyes Only",
        "effective_date": date(2076, 11, 1),
        "document_status": "ACTIVE",
    },
}

# Chunk-size experiment configurations
EXPERIMENTS: list[tuple[int, str]] = [
    (512, "vault_documents_512"),
    (1024, "vault_documents_1024"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc_slug(filepath: Path) -> str:
    """
    Convert filepath stem to a URL-safe lowercase slug.
    e.g. "Doc02_Radiation_Sickness_Symptom_Guide.docx" → "doc02_radiation_sickness_symptom_guide"
    """
    return filepath.stem.lower().replace(" ", "_").replace("-", "_")


def _build_base_metadata(filepath: Path, doc_slug: str) -> DocumentChunk | None:
    """
    Look up the metadata registry and return a DocumentChunk template.
    Returns None if the document is not in the registry.
    """
    meta = DOCUMENT_METADATA.get(doc_slug)
    if meta is None:
        logger.warning(
            "'%s' (slug: '%s') not in DOCUMENT_METADATA — skipping.", filepath.name, doc_slug
        )
        return None

    return DocumentChunk(
        chunk_id="placeholder",
        text="",
        source_document=filepath.stem,
        section_header="",
        department=meta["department"],  # type: ignore[arg-type]
        access_level=meta["access_level"],  # type: ignore[arg-type]
        effective_date=meta.get("effective_date"),
        document_status=meta["document_status"],  # type: ignore[arg-type]
        content_type="narrative",
        chunk_index=0,
        total_chunks=1,
        token_count=0,
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def ingest_document(
    filepath: Path,
    chunk_size: int,
    collection_name: str,
    output_dir: Path,
    export_dir: Path | None = None,
    dry_run: bool = False,
) -> list[DocumentChunk]:
    """
    Run the full Docling pipeline for a single document.
    Returns the list of DocumentChunk objects produced.
    """
    doc_slug = _make_doc_slug(filepath)
    base_meta = _build_base_metadata(filepath, doc_slug)
    if base_meta is None:
        return []

    logger.info("[INGEST] %s  (slug=%s, chunk_size=%d)", filepath.name, doc_slug, chunk_size)

    chunks = chunk_document_docling(
        doc_path=filepath,
        max_tokens=chunk_size,
        doc_slug=doc_slug,
        base_metadata=base_meta,
        export_dir=export_dir,
    )

    if not chunks:
        logger.warning("[INGEST] %s produced 0 chunks — skipping persist.", filepath.name)
        return []

    if not dry_run:
        persist_to_chroma(chunks, collection_name=collection_name)
        write_json_backup(chunks, output_dir=str(output_dir))

    return chunks


def run_pipeline(
    data_dir: Path,
    chunk_size: int,
    collection_name: str,
    output_dir: Path,
    export_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Ingest all supported documents in data_dir.
    Returns {filename: chunk_count} summary.
    """
    raw_files = sorted(data_dir.glob("*"))
    supported = {".pdf", ".docx", ".md", ".txt"}
    docs = [f for f in raw_files if f.suffix.lower() in supported]

    logger.info(
        "=== Vault-Tec Ingestion (Docling) ===  collection='%s'  chunk_size=%d  docs=%d",
        collection_name,
        chunk_size,
        len(docs),
    )

    summary: dict[str, int] = {}
    for doc_path in docs:
        chunks = ingest_document(
            filepath=doc_path,
            chunk_size=chunk_size,
            collection_name=collection_name,
            output_dir=output_dir,
            export_dir=export_dir,
            dry_run=dry_run,
        )
        summary[doc_path.name] = len(chunks)

    total = sum(summary.values())
    logger.info("=== Pipeline complete: %d docs → %d chunks ===", len(summary), total)
    _log_summary_table(summary)
    return summary


def _log_summary_table(summary: dict[str, int]) -> None:
    logger.info("%-60s %s", "Document", "Chunks")
    logger.info("-" * 70)
    for name, count in sorted(summary.items()):
        logger.info("%-60s %d", name, count)
    logger.info("-" * 70)
    logger.info("%-60s %d", "TOTAL", sum(summary.values()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest",
        description="Vault-Tec document ingestion pipeline (Docling-based).",
    )
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/processed/docling_exports"),
        help="Directory to save Docling markdown exports (AC9 audit).",
    )
    p.add_argument("--collection", type=str, default="vault_documents")
    p.add_argument(
        "--experiment", action="store_true", help="Run both 512 and 1024 experiments sequentially."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but do not write to ChromaDB or disk.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not args.data_dir.exists():
        logger.error("--data-dir '%s' does not exist.", args.data_dir)
        sys.exit(1)

    export_dir = args.export_dir if not args.dry_run else None

    if args.experiment:
        for size, coll in EXPERIMENTS:
            logger.info("\n" + "=" * 60)
            logger.info("EXPERIMENT: chunk_size=%d  collection=%s", size, coll)
            logger.info("=" * 60)
            run_pipeline(
                data_dir=args.data_dir,
                chunk_size=size,
                collection_name=coll,
                output_dir=args.output_dir / f"exp_{size}",
                export_dir=export_dir,
                dry_run=args.dry_run,
            )
    else:
        run_pipeline(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            collection_name=args.collection,
            output_dir=args.output_dir,
            export_dir=export_dir,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
