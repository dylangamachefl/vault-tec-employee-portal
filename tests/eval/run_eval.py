import argparse
import json
import logging
import time
import sys
from pathlib import Path

# Fix path to import src
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
# Suppress the huggingface hub token warning as well, as we rely on public embedding model
warnings.filterwarnings("ignore", module="huggingface_hub.utils._http")

from src.pipelines.retrieval_chain import VaultRetriever, QueryInput
from src.config import settings

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Constants
INTER_CALL_DELAY_SECONDS = 3  # Increased delay between our own retrieval calls
COLLECTIONS = ["vault_documents_256", "vault_documents_512", "vault_documents_1024"]
WEIGHTS = {
    "faithfulness": 0.35, 
    "answer_relevancy": 0.25,
    "context_precision": 0.25, 
    "context_recall": 0.15
}

# RAGAS processes metrics concurrently even with max_workers=1 in RunConfig. 
# We'll need a huge delay and custom patching if we want to run all 37 at once.
# To safely evaluate under the Gemini 15K TPM limit without crashing, 
# we'll evaluate 1 pair at a time and manually sleep 15 seconds.
INTER_EVAL_DELAY_SECONDS = 15

# Directories
EVAL_DIR = Path("tests/eval")
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
QA_PAIRS_FILE = EVAL_DIR / "golden_qa_pairs.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation(max_samples: int | None = None) -> None:
    # Set up RAGAS wrappers with high retries for rate limits
    logger.info("Setting up RAGAS Langchain wrappers...")
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it", 
        google_api_key=settings.google_api_key,
        max_retries=10,
        timeout=120.0
    )
    ragas_llm = LangchainLLMWrapper(llm)
    
    emb = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=settings.google_api_key
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(emb)
    
    # Configure RAGAS to run strictly sequentially to avoid hitting 15k TPM / 30 RPM limits
    # max_workers=1 forces sequential evaluation. max_wait increases retry patience.
    run_config = RunConfig(max_workers=1, max_wait=180)
    
    # Load dataset
    with open(QA_PAIRS_FILE, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
        logger.info(f"Running PILOT mode with {max_samples} samples.")
        
    all_collection_results = {}
    
    for collection_name in COLLECTIONS:
        logger.info(f"--- Evaluating collection: {collection_name} ---")
        retriever = VaultRetriever(collection_name=collection_name)
        
        data_rows = []
        
        for idx, qa in enumerate(qa_pairs):
            logger.info(f"[{collection_name}] Query {idx+1}/{len(qa_pairs)}: {qa['id']}")
            
            response = retriever.query(QueryInput(query=qa["question"], collection_name=collection_name, top_k=5))
            
            # Pack for RAGAS. Expected dictionary format for standard Dataset parsing in Ragas
            data_rows.append({
                "question": qa["question"],
                "answer": response.answer,
                "contexts": response.retrieved_chunks,
                "ground_truth": qa["ground_truth"]
            })
            
            time.sleep(INTER_CALL_DELAY_SECONDS)
            
        # Evaluate item by item to avoid rate limit crashes
        # RAGAS batches natively, which blows up the 15K TPM quota instantly.
        all_results = []
        for i in range(len(data_rows)):
            single_ds = Dataset.from_list([data_rows[i]])
            logger.info(f"[{collection_name}] Evaluating item {i+1}/{len(data_rows)}...")
            
            # Wrap in retry block to be safe
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    res = evaluate(
                        dataset=single_ds,
                        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                        llm=ragas_llm,
                        embeddings=ragas_embeddings,
                        run_config=run_config
                    )
                    all_results.append(res.to_pandas().iloc[0].to_dict())
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, sleeping 60s... ({e})")
                        time.sleep(60)
                    else:
                        logger.error(f"Failed to evaluate item {i+1} after 3 attempts.")
                        all_results.append({
                            "faithfulness": 0.0, "answer_relevancy": 0.0, 
                            "context_precision": 0.0, "context_recall": 0.0
                        })
            
            logger.info(f"[{collection_name}] Sleeping for {INTER_EVAL_DELAY_SECONDS}s to respect quotas...")
            time.sleep(INTER_EVAL_DELAY_SECONDS)
            
        import pandas as pd
        df = pd.DataFrame(all_results)
        res_dict = df.to_dict(orient="records")
        # Compute averages
        avg_scores = {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
            "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
            "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
            "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
        }
        
        out_file = RESULTS_DIR / f"eval_results_{collection_name.split('_')[-1]}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "collection": collection_name,
                "averages": avg_scores,
                "detailed": res_dict
            }, f, indent=2)
            
        all_collection_results[collection_name] = avg_scores
        logger.info(f"[{collection_name}] Saved to {out_file.name}")
        
    # Summarize and determine winner
    summary_data = {
        "candidate_scores": {},
        "composite_scores": {}
    }
    
    best_collection = None
    best_score = -1.0
    
    for collection_name, scores in all_collection_results.items():
        comp = (
            scores["faithfulness"] * WEIGHTS["faithfulness"] +
            scores["answer_relevancy"] * WEIGHTS["answer_relevancy"] +
            scores["context_precision"] * WEIGHTS["context_precision"] +
            scores["context_recall"] * WEIGHTS["context_recall"]
        )
        chunk_size = collection_name.split("_")[-1]
        
        summary_data["candidate_scores"][chunk_size] = scores
        summary_data["composite_scores"][chunk_size] = comp
        
        if comp > best_score:
            best_score = comp
            best_collection = collection_name
        elif abs(comp - best_score) <= 0.02:
            # tie break - prefer smaller chunk size.
            current_cs = int(chunk_size)
            best_cs = int(best_collection.split("_")[-1])
            if current_cs < best_cs:
                best_score = comp
                best_collection = collection_name
                
    summary_data["winning_collection"] = best_collection
    summary_data["canonical_collection_name"] = "vault_documents"
    
    summary_file = RESULTS_DIR / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # Print formatted table
    print("\n" + "="*80)
    print(f"RAGAS EVALUATION SUMMARY (N={max_samples if max_samples else len(qa_pairs)} pairs)")
    print("="*80)
    print(f"{'Metric':<20} | {'256':<10} | {'512':<10} | {'1024':<10}")
    print("-" * 60)
    
    for metric in WEIGHTS.keys():
        s256 = summary_data["candidate_scores"]["256"][metric]
        s512 = summary_data["candidate_scores"]["512"][metric]
        s1024 = summary_data["candidate_scores"]["1024"][metric]
        print(f"{metric:<20} | {s256:<10.4f} | {s512:<10.4f} | {s1024:<10.4f}")
        
    print("-" * 60)
    c256 = summary_data["composite_scores"]["256"]
    c512 = summary_data["composite_scores"]["512"]
    c1024 = summary_data["composite_scores"]["1024"]
    print(f"{'COMPOSITE':<20} | {c256:<10.4f} | {c512:<10.4f} | {c1024:<10.4f}")
    print("="*80)
    print(f"WINNING COLLECTION:    {summary_data['winning_collection']}")
    print(f"CANONICAL COLLECTION:  {summary_data['canonical_collection_name']}")
    print("="*80 + "\n")


def write_canonical(summary_file: Path) -> None:
    if not summary_file.exists():
        logger.error(f"Cannot write canonical: {summary_file} missing. Run eval first.")
        return
        
    with open(summary_file, "r") as f:
        summary_data = json.load(f)
        
    winner = summary_data["winning_collection"]
    winner_chunk_size = winner.split("_")[-1]
    
    logger.info(f"Writing canonical collection from {winner}...")
    
    # Read the json backup from data/processed/exp_{size}
    backup_dir = Path("data/processed") / f"exp_{winner_chunk_size}"
    
    # Read all chunks
    all_chunks = []
    for f in backup_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as jf:
            all_chunks.append(json.load(jf))
            
    logger.info(f"Loaded {len(all_chunks)} chunks from {backup_dir}")
    
    import chromadb
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    
    # Recreate the collection
    try:
        client.delete_collection("vault_documents")
        logger.info("Deleted existing canonical 'vault_documents' collection")
    except Exception:
        pass
        
    coll = client.create_collection("vault_documents")
    
    ids = [c["metadata"]["chunk_id"] for c in all_chunks]
    docs = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]
    
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        coll.upsert(
            ids=ids[i:end],
            documents=docs[i:end],
            metadatas=metas[i:end]
        )
        
    logger.info(f"Verification: 'vault_documents' has {coll.count()} chunks.")
    print(f"\nvault_documents written: {coll.count()} chunks from {winner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 3 samples to verify pipeline")
    parser.add_argument("--write-canonical", action="store_true", help="Write winner to canonical collection")
    args = parser.parse_args()
    
    if args.write_canonical:
        write_canonical(RESULTS_DIR / "eval_summary.json")
    else:
        run_evaluation(max_samples=3 if args.pilot else None)
