from bielik_hallu.dataset.label import judge_known, label_row


def test_judge_known_all_correct():
    answers = ["a", "b", "c"]
    assert judge_known("X", answers, judge_fn=lambda e, a: True) is True


def test_judge_known_one_wrong():
    answers = ["a", "b", "c"]
    calls = {"n": 0}
    def jf(e, a):
        calls["n"] += 1
        return calls["n"] != 2  # second one is wrong
    assert judge_known("X", answers, judge_fn=jf) is False


def test_label_row_rules():
    assert label_row("FABRICATED", all_correct=False) == 1
    assert label_row("UNKNOWN_REAL", all_correct=False) == 1
    assert label_row("KNOWN", all_correct=True) == 0
    assert label_row("KNOWN", all_correct=False) == 1
