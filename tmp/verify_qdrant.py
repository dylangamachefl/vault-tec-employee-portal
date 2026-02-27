import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import settings
from src.pipelines.persist import get_qdrant_client


def main():
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    print(f"Checking collection: {collection_name}")

    try:
        response = client.scroll(collection_name=collection_name, limit=5, with_payload=True)
        points, _ = response

        if not points:
            print("ERROR: No points found in collection.")
            sys.exit(1)

        print(f"Found {len(points)} points. Checking first one...")

        payload = points[0].payload
        print("\nPayload:")
        for k, v in payload.items():
            print(f"  {k}: {v}")

        required_fields = ["source", "date", "access_level", "department"]
        missing = []
        for field in required_fields:
            if field not in payload:
                missing.append(field)

        if missing:
            print(f"\nERROR: Missing required fields: {missing}")
            sys.exit(1)
        else:
            print("\nSUCCESS: All required fields present.")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
