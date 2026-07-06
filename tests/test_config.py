from bielik_hallu import config


def test_config_has_three_conditions():
    assert config.CONDITIONS == ("KNOWN", "UNKNOWN_REAL", "FABRICATED")


def test_prompt_template_formats():
    assert config.PROMPT_TEMPLATE.format(entity="X") == "Kim jest X? Odpowiedz jednym zdaniem."
