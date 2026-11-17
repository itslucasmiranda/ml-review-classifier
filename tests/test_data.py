import hashlib
import json
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from review_classifier import data
from review_classifier.data import Review


def synthetic_reviews():
    return [Review(f'{domain} example {label} number {index}', label, domain, label * 10 + index + 1)
            for domain in data.MEMBERS for label in (0, 1) for index in range(10)]


def archive_fixture(monkeypatch):
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        for domain, member in data.MEMBERS.items():
            archive.writestr(member, ''.join(f'{row.text}\t{row.label}\n' for row in synthetic_reviews() if row.domain == domain))
        archive.writestr('__MACOSX/irrelevant', b'\xff\xfe')
    raw = buffer.getvalue()
    monkeypatch.setattr(data, 'ARCHIVE_SHA256', hashlib.sha256(raw).hexdigest())
    return raw


def test_parser_preserves_quotes_tabs_unicode_separators_and_original_text():
    rows = data.parse_reviews('"Great!"\t1\r\nA\u0085fine\u2028movie\twith tabs\t0\n', 'imdb')
    assert [r.text for r in rows] == ['"Great!"', 'A\u0085fine\u2028movie\twith tabs']
    assert [r.source_line for r in rows] == [1, 2]


@pytest.mark.parametrize('contents', ['', 'no label', 'text\t2', 'text\ttrue', '\t1', '\n', 'ok\t1\n\nother\t0'])
def test_parser_rejects_invalid_rows(contents):
    with pytest.raises(ValueError):
        data.parse_reviews(contents, 'amazon')


@pytest.mark.parametrize('label', [-1, 2, True, '1', 1.0])
def test_review_rejects_nonbinary_or_noninteger_labels(label):
    with pytest.raises(ValueError):
        Review('text', label, 'amazon', 1)


def test_deduplication_is_global_and_removes_conflicting_groups():
    rows = [Review(' ＧＲＥＡＴ  item ', 1, 'yelp', 2), Review('great item', 1, 'amazon', 1),
            Review('Bad', 0, 'imdb', 1), Review('BAD', 1, 'amazon', 3),
            Review('Unique!', 0, 'yelp', 4)]
    cleaned, removed = data.deduplicate(rows)
    assert {(r.text, r.domain) for r in cleaned} == {('great item', 'amazon'), ('Unique!', 'yelp')}
    assert removed == {'duplicate_rows_removed': 1, 'conflicting_rows_removed': 2, 'conflicting_groups_removed': 1}
    assert data.deduplicate(list(reversed(rows))) == (cleaned, removed)


def test_split_is_disjoint_complete_stratified_and_order_independent():
    rows = synthetic_reviews()
    splits = data.split_reviews(rows)
    assert splits == data.split_reviews(list(reversed(rows)))
    assert {k: len(v) for k, v in splits.items()} == {'train': 36, 'validation': 12, 'test': 12}
    sets = [{r.text_id for r in part} for part in splits.values()]
    assert len(set.union(*sets)) == len(rows)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for name, per_group in [('train', 6), ('validation', 2), ('test', 2)]:
        assert set(data.counts(splits[name])['domain_labels'].values()) == {per_group}
    assert splits != data.split_reviews(rows, seed=43)


@pytest.mark.parametrize('seed', [-1, 2**32, True, 1.5])
def test_split_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):
        data.split_reviews(synthetic_reviews(), seed)


def test_split_rejects_duplicate_or_small_input():
    rows = synthetic_reviews()
    for sample in ([], rows + [rows[0]], rows[:4]):
        with pytest.raises(ValueError):
            data.split_reviews(sample)


def test_download_verifies_bytes_and_reuses_cache_without_network(monkeypatch, tmp_path):
    raw = archive_fixture(monkeypatch)
    calls = []

    @contextmanager
    def download(url, timeout):
        calls.append((url, timeout))
        yield BytesIO(raw)

    monkeypatch.setattr(data, 'urlopen', download)
    path = tmp_path / 'cache/data.zip'
    assert data.obtain_archive(path) == raw
    assert data.obtain_archive(path) == raw
    assert calls == [(data.DATASET_URL, 30)]
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        data.obtain_archive(path)
    assert len(calls) == 1


def test_failed_download_never_saves_unverified_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'urlopen', lambda *args, **kwargs: BytesIO(b'invalid archive'))
    with pytest.raises(ValueError, match='checksum'):
        data.obtain_archive(tmp_path / 'data.zip')
    assert not (tmp_path / 'data.zip').exists()


def test_download_size_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'MAX_ARCHIVE_BYTES', 10)
    monkeypatch.setattr(data, 'urlopen', lambda *args, **kwargs: BytesIO(b'x' * 11))
    with pytest.raises(ValueError, match='size limit'):
        data.obtain_archive(tmp_path / 'data.zip')


def test_read_archive_checks_integrity_before_parsing():
    with pytest.raises(ValueError, match='checksum'):
        data.read_archive(b'not a zip')


def test_prepare_writes_reproducible_splits_and_auditable_hashes(monkeypatch, tmp_path):
    archive = tmp_path / 'reviews.zip'
    archive.write_bytes(archive_fixture(monkeypatch))
    first, second = tmp_path / 'first', tmp_path / 'second'
    report = data.prepare_dataset(archive, first)
    assert report == data.prepare_dataset(archive, second)
    assert report['raw']['rows'] == report['cleaned']['rows'] == 60
    assert report['split_overlap_count'] == 0
    assert json.loads((first / 'audit.json').read_text()) == report
    for name, metadata in report['splits'].items():
        raw = (first / f'{name}.jsonl').read_bytes()
        assert raw == (second / f'{name}.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata['sha256']
        assert len([json.loads(line) for line in raw.splitlines()]) == metadata['rows']
    with pytest.raises(FileExistsError):
        data.prepare_dataset(archive, first)
    assert json.loads((first / 'audit.json').read_text()) == report


def test_existing_output_is_rejected_before_download(monkeypatch, tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('keep')
    monkeypatch.setattr(data, 'obtain_archive', lambda _: pytest.fail('Should fail before download'))
    with pytest.raises(FileExistsError):
        data.prepare_dataset(tmp_path / 'absent.zip', output)
    assert marker.read_text() == 'keep'


def test_invalid_rows_leave_no_partial_output(monkeypatch, tmp_path):
    raw = archive_fixture(monkeypatch)
    path = tmp_path / 'data.zip'
    path.write_bytes(raw)
    monkeypatch.setattr(data, 'read_archive', lambda _: [])
    with pytest.raises(ValueError):
        data.prepare_dataset(path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
