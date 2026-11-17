import json

import pytest

from review_classifier import prepare_data


def test_cli_reports_failure_without_success_output(monkeypatch, capsys, tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    monkeypatch.setattr('sys.argv', ['prepare_data', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        prepare_data.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert 'already exists' in captured.err
    assert captured.out == ''
