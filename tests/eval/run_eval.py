import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Fix path to import src
sys.path.insert(0, ".")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
# Suppress the huggingface hub token warning as well, as we rely on public embedding model
warnings.filterwarnings("ignore", module="huggingface_hub.utils._http")

import httpx  # noqa: E402
from datasets import Dataset  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from ragas import RunConfig, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import settings  # noqa: E402
from src.pipelines.retrieval_chain import VaultRetriever  # noqa: E402

# Constants
INTER_CALL_DELAY_SECONDS = 15  # Increased to avoid Google AI Studio free tier 15K TPM limit
EVAL_TOP_K = 5  # Increased to 5 per user request
COLLECTIONS = ["vault_documents_256", "vault_documents_512", "vault_documents_1024"]
WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevancy": 0.25,
    "context_precision": 0.25,
    "context_recall": 0.15,
}

# Directories
EVAL_DIR = Path("tests/eval")
QA_PAIRS_FILE = EVAL_DIR / "golden_qa_pairs.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


def check_ollama_running() -> None:
    """Pre-flight: confirm Ollama is up and required models are pulled."""
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        required = ["gemma3:4b-it-qat", "nomic-embed-text"]
        missing = [m for m in required if not any(m in name for name in models)]
        if missing:
            raise RuntimeError(
                f"Required Ollama models not found: {missing}. " f"Run: ollama pull <model>"
            )
        print(f"✓ Ollama running. Required models confirmed: {models}")
    except httpx.ConnectError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")


class LoggingChatOllama(ChatOllama):
    """ChatOllama with basic call logging to show progress even if the progress bar is stuck."""

    def _generate(self, messages, *args, **kwargs):
        # logging prompt length helps diagnose context window issues
        total_chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
        logger.info(f"  [Judge] Calling LLM (sync, ~{total_chars} chars)...")
        return super()._generate(messages, *args, **kwargs)

    async def _agenerate(self, messages, *args, **kwargs):
        total_chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
        logger.info(f"  [Judge] Calling LLM (async, ~{total_chars} chars)...")
        return await super()._agenerate(messages, *args, **kwargs)


def build_ragas_wrappers():
    """Return (ragas_llm, ragas_embeddings) backed by local Ollama."""
    ragas_llm = LangchainLLMWrapper(
        LoggingChatOllama(
            model="gemma3:4b-it-qat", base_url="http://localhost:11434", timeout=300, num_ctx=8192
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    )
    return ragas_llm, ragas_embeddings


def load_qa_pairs(max_samples: int | None = None) -> list[dict]:
    with open(QA_PAIRS_FILE, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    if max_samples:
        return qa_pairs[:max_samples]
    return qa_pairs


def run_generation_phase(
    collections: list[str], run_dir: Path, max_samples: int | None = None
) -> None:
    """Phase A: Generate answers via Google AI Studio and cache them to disk."""
    qa_pairs = load_qa_pairs(max_samples)
    run_dir.mkdir(parents=True, exist_ok=True)

    for collection_name in collections:
        logger.info(f"--- Generating answers for collection: {collection_name} ---")
        retriever = VaultRetriever(collection_name=collection_name)
        data_rows = []

        chunk_size = collection_name.split("_")[-1]
        log_file = run_dir / f"generation_log_{chunk_size}.txt"

        with open(log_file, "w", encoding="utf-8") as log_fh:
            _divider = "=" * 100

            log_fh.write(f"GENERATION LOG — {collection_name}\n")
            log_fh.write(f"Run dir: {run_dir}\n")
            log_fh.write(f"QA pairs: {len(qa_pairs)}  |  top_k: {EVAL_TOP_K}\n")
            log_fh.write(f"{_divider}\n\n")

            for idx, qa in enumerate(qa_pairs):
                logger.info(f"[{collection_name}] Query {idx+1}/{len(qa_pairs)}: {qa['id']}")

                t0 = time.time()

                # Retrieve chunks separately so we can log them before the LLM call
                chunks = retriever.retrieve(
                    query=qa["question"],
                    top_k=EVAL_TOP_K,
                    access_level_filter=None,
                )

                # --- Log header ---
                log_fh.write(f"[{idx+1}/{len(qa_pairs)}] ID: {qa['id']}\n")
                log_fh.write(f"QUESTION : {qa['question']}\n")
                log_fh.write(f"GROUND TRUTH: {qa['ground_truth']}\n\n")

                # --- Log retrieved chunks ---
                log_fh.write(f"RETRIEVED CHUNKS ({len(chunks)}):\n")
                for ci, chunk in enumerate(chunks, 1):
                    meta = chunk.metadata
                    log_fh.write(
                        f"  [{ci}] source={meta.get('source_document','?')} | "
                        f"section={meta.get('section_header') or 'N/A'} | "
                        f"type={meta.get('content_type','?')} | "
                        f"tokens={meta.get('token_count','?')}\n"
                    )
                    log_fh.write(f"  TEXT:\n{chunk.content}\n\n")

                # --- Call LLM and log answer ---
                response = retriever.generate(query=qa["question"], chunks=chunks)
                t1 = time.time()

                log_fh.write(f"LLM ANSWER ({t1 - t0:.2f}s):\n{response.answer}\n")
                log_fh.write(f"\n{_divider}\n\n")
                log_fh.flush()

                logger.info(f"  -> Generated in {t1 - t0:.2f}s")

                data_rows.append(
                    {
                        "question": qa["question"],
                        "answer": response.answer,
                        "retrieved_chunks": response.retrieved_chunks,
                        "ground_truth": qa["ground_truth"],
                        "collection": collection_name,
                    }
                )
                logger.info(f"  -> Sleeping for {INTER_CALL_DELAY_SECONDS}s...")
                time.sleep(INTER_CALL_DELAY_SECONDS)

        out_file = run_dir / f"generated_answers_{chunk_size}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data_rows, f, indent=2, default=str)

        logger.info(f"[{collection_name}] Saved generated answers to {out_file.name}")
        logger.info(f"[{collection_name}] Detailed log saved to {log_file.name}")


def run_scoring_phase(
    collections: list[str], run_dir: Path, is_pilot: bool = False, print_summary: bool = True
) -> None:
    """Phase B: Load cached answers and evaluate them using standard evaluation."""
    check_ollama_running()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up RAGAS Ollama wrappers...")
    ragas_llm, ragas_embeddings = build_ragas_wrappers()

    # Using 10-minute timeout and 1 worker for collection 512 to ensure stability.
    # 512 chunks are larger and can be slow/resource-heavy for local LLMs.
    run_config = RunConfig(max_workers=1, timeout=600)

    all_collection_results = {}

    for collection_name in collections:
        chunk_size = collection_name.split("_")[-1]
        results_file = run_dir / f"eval_results_{chunk_size}.json"

        # Skip if results already exist (unless pilot)
        if results_file.exists() and not is_pilot:
            logger.info(f"--- Skipping {collection_name} (results already exist) ---")
            with open(results_file, encoding="utf-8") as f:
                res_data = json.load(f)
                all_collection_results[collection_name] = res_data.get("averages", {})
            continue

        logger.info(
            f"--- Evaluating collection: {collection_name} ({'Pilot' if is_pilot else 'Full'}) ---"
        )
        answers_file = run_dir / f"generated_answers_{chunk_size}.json"

        if not answers_file.exists():
            raise FileNotFoundError(
                f"Missing {answers_file.name} — run with --generate-answers first."
            )

        with open(answers_file, encoding="utf-8") as f:
            data_rows = json.load(f)

        if is_pilot:
            data_rows = data_rows[:3]
            logger.info(f"[{collection_name}] Running PILOT scoring with {len(data_rows)} samples.")

        # Standardize 'contexts' for RAGAS dataset
        for row in data_rows:
            row["contexts"] = row["retrieved_chunks"]

        logger.info(f"--- Evaluating collection: {collection_name} ({len(data_rows)} items) ---")

        dataset = Dataset.from_list(data_rows)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=run_config,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"RAGAS evaluation error (attempt {attempt+1}): {e}. Retrying..."
                    )
                    time.sleep(10)
                else:
                    logger.error(f"[{collection_name}] Failed after {max_retries} attempts: {e}")
                    raise

        df = res.to_pandas()
        res_dict = df.to_dict(orient="records")

        avg_scores = {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
            "answer_relevancy": float(df["answer_relevancy"].mean())
            if "answer_relevancy" in df
            else 0.0,
            "context_precision": float(df["context_precision"].mean())
            if "context_precision" in df
            else 0.0,
            "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
        }

        out_file = run_dir / f"eval_results_{chunk_size}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {"collection": collection_name, "averages": avg_scores, "detailed": res_dict},
                f,
                indent=2,
                default=str,  # handle any non-serialisable floats (NaN → "nan")
            )

        all_collection_results[collection_name] = avg_scores
        logger.info(f"[{collection_name}] Saved to {out_file.name}")

    if not print_summary:
        return

    # Summarize and determine winner
    summary_data = {"candidate_scores": {}, "composite_scores": {}}

    best_collection = None
    best_score = -1.0

    for collection_name, scores in all_collection_results.items():
        comp = (
            scores["faithfulness"] * WEIGHTS["faithfulness"]
            + scores["answer_relevancy"] * WEIGHTS["answer_relevancy"]
            + scores["context_precision"] * WEIGHTS["context_precision"]
            + scores["context_recall"] * WEIGHTS["context_recall"]
        )
        chunk_size = collection_name.split("_")[-1]

        summary_data["candidate_scores"][chunk_size] = scores
        summary_data["composite_scores"][chunk_size] = comp

        if comp > best_score:
            best_score = comp
            best_collection = collection_name
        elif abs(comp - best_score) <= 0.02:
            # tie break — prefer smaller chunk size
            current_cs = int(chunk_size)
            best_cs = int(best_collection.split("_")[-1])
            if current_cs < best_cs:
                best_score = comp
                best_collection = collection_name

    summary_data["winning_collection"] = best_collection
    summary_data["canonical_collection_name"] = "vault_documents"

    summary_file = run_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Print formatted table
    n = len(data_rows)  # Using size of last collection scored
    print("\n" + "=" * 80)
    print(f"RAGAS EVALUATION SUMMARY (N={n} pairs)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'256':<10} | {'512':<10} | {'1024':<10}")
    print("-" * 60)

    for metric in WEIGHTS.keys():
        s256 = summary_data["candidate_scores"].get("256", {}).get(metric, 0)
        s512 = summary_data["candidate_scores"].get("512", {}).get(metric, 0)
        s1024 = summary_data["candidate_scores"].get("1024", {}).get(metric, 0)
        print(f"{metric:<20} | {s256:<10.4f} | {s512:<10.4f} | {s1024:<10.4f}")

    print("-" * 60)
    c256 = summary_data["composite_scores"].get("256", 0)
    c512 = summary_data["composite_scores"].get("512", 0)
    c1024 = summary_data["composite_scores"].get("1024", 0)
    print(f"{'COMPOSITE':<20} | {c256:<10.4f} | {c512:<10.4f} | {c1024:<10.4f}")
    print("=" * 80)
    print(f"WINNING COLLECTION:    {summary_data.get('winning_collection')}")
    print(f"CANONICAL COLLECTION:  {summary_data.get('canonical_collection_name')}")
    print("=" * 80 + "\n")


def write_canonical(summary_file: Path) -> None:
    if not summary_file.exists():
        logger.error(f"Cannot write canonical: {summary_file} missing. Run eval first.")
        return

    with open(summary_file) as f:
        summary_data = json.load(f)

    winner = summary_data.get("winning_collection")
    if not winner:
        logger.error("No winning collection found in summary!")
        return

    winner_chunk_size = winner.split("_")[-1]

    logger.info(f"Writing canonical collection from {winner}...")

    backup_dir = Path("data/processed") / f"exp_{winner_chunk_size}"

    all_chunks = []
    for f in backup_dir.glob("*.json"):
        with open(f, encoding="utf-8") as jf:
            all_chunks.append(json.load(jf))

    logger.info(f"Loaded {len(all_chunks)} chunks from {backup_dir}")

    import chromadb

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    try:
        client.delete_collection("vault_documents")
        logger.info("Deleted existing canonical 'vault_documents' collection")
    except Exception:
        pass

    coll = client.create_collection("vault_documents")

    ids = [c["metadata"]["chunk_id"] for c in all_chunks]
    docs = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        coll.upsert(ids=ids[i:end], documents=docs[i:end], metadatas=metas[i:end])

    logger.info(f"Verification: 'vault_documents' has {coll.count()} chunks.")
    print(f"\nvault_documents written: {coll.count()} chunks from {winner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Phase A: Generate answers via Google AI Studio and cache them",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Phase B: Only run Ollama RAGAS scoring on cached answers",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 3 samples against vault_documents_512 using cached answers (--score-only behaviour)",
    )
    parser.add_argument(
        "--write-canonical", action="store_true", help="Write winner to canonical collection"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Specify a single collection to evaluate (e.g., vault_documents_256).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="Name of the run folder (created in tests/eval/runs/)",
    )
    args = parser.parse_args()

    run_dir = EVAL_DIR / "runs" / args.run_name

    if args.write_canonical:
        write_canonical(run_dir / "eval_summary.json")
    elif args.pilot and args.generate_answers:
        run_generation_phase([args.collection or "vault_documents_512"], run_dir, max_samples=3)
        logger.info("Pilot generation phase complete. Cached answers saved. Proceed to --pilot.")
    elif args.pilot:
        run_scoring_phase(
            [args.collection or "vault_documents_512"], run_dir, is_pilot=True, print_summary=False
        )
        logger.info("Pilot run complete. Review results above, then run without --pilot.")
    else:
        collections_to_run = [args.collection] if args.collection else COLLECTIONS
        if args.generate_answers:
            run_generation_phase(collections_to_run, run_dir)
            logger.info("Generation phase complete. Cached answers saved. Proceed to --score-only.")
        elif args.score_only:
            run_scoring_phase(collections_to_run, run_dir)
        else:
            # Fallback behaviour: combined run
            run_generation_phase(collections_to_run, run_dir)
            run_scoring_phase(collections_to_run, run_dir)
