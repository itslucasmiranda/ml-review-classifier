import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
ARCHIVE_SHA256 = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"
MAX_ARCHIVE_BYTES = 2_000_000
MEMBERS = {
    "amazon": "sentiment labelled sentences/amazon_cells_labelled.txt",
    "imdb": "sentiment labelled sentences/imdb_labelled.txt",
    "yelp": "sentiment labelled sentences/yelp_labelled.txt",
}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Review:
    text: str
    label: int
    domain: str
    source_line: int

    def __post_init__(self):
        if not isinstance(self.text, str) or not normalize_text(self.text):
            raise ValueError("Review text must not be empty")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("Label must be integer 0 or 1")
        if self.domain not in MEMBERS:
            raise ValueError("Unknown domain")
        if type(self.source_line) is not int or self.source_line < 1:
            raise ValueError("Source line must be a positive integer")

    @property
    def text_id(self) -> str:
        return hashlib.sha256(normalize_text(self.text).encode("utf-8")).hexdigest()


def verify_archive(raw: bytes) -> None:
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Dataset checksum mismatch; the archive must match the documented version")


def obtain_archive(path: Path) -> bytes:
    if path.exists():
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Dataset archive exceeds the size limit")
        raw = path.read_bytes()
        verify_archive(raw)
        return raw
    with urlopen(DATASET_URL, timeout=30) as response:
        raw = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Dataset archive exceeds the size limit")
    verify_archive(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
    return raw


def parse_reviews(contents: str, domain: str) -> list[Review]:
    # Only LF separates records; Unicode separators may occur inside a sentence.
    lines = contents.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    reviews = []
    for number, line in enumerate(lines, start=1):
        fields = line.removesuffix("\r").rsplit("\t", 1)
        if len(fields) != 2 or fields[1] not in ("0", "1"):
            raise ValueError(f"Invalid row in {domain} at line {number}")
        reviews.append(Review(fields[0], int(fields[1]), domain, number))
    if not reviews:
        raise ValueError(f"No reviews in {domain}")
    return reviews


def read_archive(raw: bytes) -> list[Review]:
    verify_archive(raw)
    reviews = []
    with ZipFile(BytesIO(raw)) as archive:
        for domain, member in MEMBERS.items():
            contents = archive.read(member).decode("utf-8-sig")
            reviews.extend(parse_reviews(contents, domain))
    return reviews


def counts(reviews: list[Review]) -> dict:
    return {
        "rows": len(reviews),
        "labels": dict(sorted(Counter(str(row.label) for row in reviews).items())),
        "domains": dict(sorted(Counter(row.domain for row in reviews).items())),
        "domain_labels": dict(sorted(Counter(f"{row.domain}:{row.label}" for row in reviews).items())),
    }


def deduplicate(reviews: list[Review]) -> tuple[list[Review], dict]:
    groups = defaultdict(list)
    for row in reviews:
        groups[row.text_id].append(row)
    kept = []
    duplicate_rows = conflict_rows = conflict_groups = 0
    for text_id in sorted(groups):
        group = groups[text_id]
        if len({row.label for row in group}) > 1:
            conflict_rows += len(group)
            conflict_groups += 1
        else:
            kept.append(min(group, key=lambda row: (row.domain, row.source_line, row.text)))
            duplicate_rows += len(group) - 1
    return kept, {"duplicate_rows_removed": duplicate_rows,
                  "conflicting_rows_removed": conflict_rows,
                  "conflicting_groups_removed": conflict_groups}


def split_reviews(reviews: list[Review], seed: int = 42) -> dict[str, list[Review]]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295")
    if not reviews or len({row.text_id for row in reviews}) != len(reviews):
        raise ValueError("Splitting requires nonempty, deduplicated input")
    strata = defaultdict(list)
    for row in reviews:
        strata[(row.domain, row.label)].append(row)
    splits = {"train": [], "validation": [], "test": []}
    for stratum in sorted(strata):
        rows = sorted(strata[stratum], key=lambda row: (
            hashlib.sha256(f"{seed}:{row.text_id}".encode()).hexdigest(), row.text_id
        ))
        if len(rows) < 5:
            raise ValueError(f"At least five unique reviews required in stratum {stratum}")
        train_end = 3 * len(rows) // 5
        validation_end = train_end + len(rows) // 5
        splits["train"].extend(rows[:train_end])
        splits["validation"].extend(rows[train_end:validation_end])
        splits["test"].extend(rows[validation_end:])
    return splits


def prepare_dataset(archive_path: Path, output: Path, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295")
    raw = obtain_archive(archive_path)
    original = read_archive(raw)
    cleaned, removed = deduplicate(original)
    splits = split_reviews(cleaned, seed)
    all_ids = [row.text_id for rows in splits.values() for row in rows]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(cleaned):
        raise ValueError("Split integrity check failed")
    report = {
        "dataset": "uci_sentiment_labelled_sentences", "source_url": DATASET_URL,
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "seed": seed,
        "normalization": "unicode_nfkc_casefold_whitespace_v1",
        "split_algorithm": "sha256_seed_text_id_rank_per_domain_label_v1",
        "requested_ratios": {"train": .6, "validation": .2, "test": .2},
        "raw": counts(original), "cleaned": counts(cleaned), "removed": removed,
        "split_overlap_count": len(all_ids) - len(set(all_ids)),
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name, rows in splits.items():
            payload = "".join(json.dumps({"id": row.text_id, "text": row.text, "label": row.label,
                                        "domain": row.domain, "source_line": row.source_line},
                                       ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
            (staging / f"{name}.jsonl").write_bytes(payload)
            report["splits"][name] = {**counts(rows), "sha256": hashlib.sha256(payload).hexdigest()}
        (staging / "audit.json").write_bytes((json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        staging.rename(output)
    return report
