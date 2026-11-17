import argparse
import json
from pathlib import Path
from urllib.error import URLError
from zipfile import BadZipFile

from review_classifier.data import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare verified review data and reproducible train/validation/test splits.")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/sentiment-labelled-sentences.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/uci-v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = prepare_dataset(args.archive, args.output, args.seed)
    except (ValueError, OSError, URLError, BadZipFile, KeyError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps({"raw_rows": report["raw"]["rows"], "cleaned_rows": report["cleaned"]["rows"],
                      "removed": report["removed"], "splits": {name: info["rows"] for name, info in report["splits"].items()},
                      "split_overlap_count": report["split_overlap_count"]}, indent=2))
    print(f"Audit: {(args.output / 'audit.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
