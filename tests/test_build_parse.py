from bielik_hallu.dataset.build import parse_correct_verdict


def test_parse_clean_true():
    assert parse_correct_verdict('{"correct": true}') is True


def test_parse_clean_false():
    assert parse_correct_verdict('{"correct": false}') is False


def test_parse_markdown_fenced():
    assert parse_correct_verdict('```json\n{"correct": true}\n```') is True


def test_parse_prose_wrapped():
    assert parse_correct_verdict('Based on the answer: {"correct": false}. Done.') is False


def test_parse_garbage_no_json():
    assert parse_correct_verdict('I think it is correct') is False


def test_parse_missing_key():
    assert parse_correct_verdict('{"verdict": true}') is False
