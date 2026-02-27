import sys

import chromadb


def main():
    client = chromadb.HttpClient(host="localhost", port=8002)
    collection = client.get_collection("vault_documents_512")

    result = collection.get(
        where={"chunk_id": "doc02_radiation_sickness_symptom_guide_003"},
        include=["documents", "metadatas"],
    )

    if not result["documents"]:
        print("FAIL: Chunk not found!")
        sys.exit(1)

    text = result["documents"][0]
    print("--- CHUNK TEXT ---")
    print(text)
    print("------------------")

    # Check 1: Pipe characters
    if "|" not in text:
        print("FAIL: No pipe characters found in chunk text.")
        sys.exit(1)
    else:
        print("PASS: Pipe characters found.")

    # Check 2: VT-MED-011
    if "VT-MED-011" not in text:
        print("FAIL: VT-MED-011 not found in chunk text.")
        sys.exit(1)
    else:
        print("PASS: VT-MED-011 found.")

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
