# Chunking Evaluation Log

## Rationale for `512` Token Chunk Size
During our RAGAS framework evaluation across multiple candidate lengths (256, 512, and 1024 tokens), we observed the most balanced and strongest performance metrics with the `512` token chunk size. 

A `512` token size provides enough contextual window to capture cohesive narrative paragraphs and policy details from the Vault-Tec documentation without diluting the semantic relevance through excessive noise, which we observed starting to occur at larger chunk sizes. Smaller chunks (256) occasionally severed important procedural context, while larger chunks (1024) degraded `context_precision` by forcing the embedder to average out too many diverse topics within a single chunk string.

The optimal vector representation aligned with `512` tokens because it maintains a high degree of faithfulness to the original document while enabling precise retriever matching.

## Final Evaluated Metrics (Chunk Size: 512)

| Metric | Score |
| :--- | :--- |
| **Faithfulness** | 0.9333 |
| **Answer Relevancy** | 0.5530 |
| **Context Precision** | 0.8059 |
| **Context Recall** | 0.9559 |
| **Composite Score** | 0.8098 |

*Metrics produced across N=34 verification queries via the RAGAS evaluation harness.*

## Conclusion

Given the empirical validation of the `512` chunk size in yielding \>0.90 scores for recall and faithfulness, it has been promoted to a hardcoded `CHUNK_SIZE` parameter within `src/config.py` along with a corresponding `CHUNK_OVERLAP` of `64`. All hardcoded magic numbers relating to size have been purged from the `HybridChunker` instantiations to ensure configuration consistency.

## Ingestion Summary

The full corpus of standard Vault-Tec administrative documentation (16 documents) was successfully ingested and parsed using `docling` following the parameter lock.
