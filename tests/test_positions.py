import pytest
from transformers import AutoTokenizer
from bielik_hallu import config
from bielik_hallu.extract.positions import (
    find_entity_last_token,
    find_entity_last_token_in_offsets,
    last_prompt_token_index,
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(config.MODEL_ID)


def test_last_prompt_token_index():
    assert last_prompt_token_index([5, 6, 7]) == 2


def test_entity_last_token_points_inside_entity(tok):
    rendered = "Kim jest Robert Lewandowski? Odpowiedz jednym zdaniem."
    idx = find_entity_last_token(tok, rendered, "Robert Lewandowski")
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    start, end = enc["offset_mapping"][idx]
    ent_start = rendered.index("Robert Lewandowski")
    ent_end = ent_start + len("Robert Lewandowski")
    assert start >= ent_start and end <= ent_end


def test_find_entity_last_token_in_offsets_pure():
    rendered = "Kim jest AB? x"
    entity = "AB"
    ent_start = rendered.index(entity)
    ent_end = ent_start + len(entity)
    # Hand-built offset_mapping simulating a model-input encoding with a
    # leading BOS-like special token at (0, 0), followed by real tokens.
    # "Kim"(0-3) " jest"(3-8) " "(9-10) "AB"(10-12) "?"(12-13) " x"(13-15)
    offset_mapping = [
        (0, 0),      # BOS-like special token, must be skipped
        (0, 3),      # "Kim"
        (3, 8),      # " jest"
        (9, 10),     # " "
        (ent_start, ent_start + 1),  # "A" (first half of entity)
        (ent_start + 1, ent_end),    # "B" (second half of entity)
        (12, 13),    # "?"
        (13, 15),    # " x"
    ]
    idx = find_entity_last_token_in_offsets(offset_mapping, rendered, entity)
    start, end = offset_mapping[idx]
    assert start >= ent_start and end <= ent_end
    assert idx == 5  # last token fully within the entity span
    assert idx != 0  # leading (0, 0) BOS-like span must be skipped


def test_leading_space_merged_single_token():
    # SentencePiece/BPE merges the preceding space into the entity's first
    # subword, so a single-token entity gets an offset that starts one char
    # before ent_start. Regression for the cities "Czym jest Warszawa?" case.
    rendered = "Czym jest Warszawa? Odpowiedz jednym zdaniem."
    entity = "Warszawa"
    ent_start = rendered.index(entity)   # 10
    ent_end = ent_start + len(entity)    # 18
    offset_mapping = [
        (0, 0),                    # BOS-like special token
        (0, 4),                    # "Czym"
        (4, 9),                    # " jest"
        (ent_start - 1, ent_end),  # " Warszawa" (leading space merged in)
        (ent_end, ent_end + 1),    # "?"
    ]
    idx = find_entity_last_token_in_offsets(offset_mapping, rendered, entity)
    assert idx == 3  # the merged " Warszawa" token is the entity's last token


def test_leading_letter_of_other_word_not_accepted():
    # A token that reaches back into a non-whitespace preceding char must be
    # rejected (guards against grabbing a neighbouring word's token).
    rendered = "xWarszawa"
    entity = "Warszawa"
    ent_start = rendered.index(entity)   # 1
    ent_end = ent_start + len(entity)
    offset_mapping = [
        (0, ent_end),  # "xWarszawa" — spills back to a letter, not whitespace
    ]
    with pytest.raises(ValueError):
        find_entity_last_token_in_offsets(offset_mapping, rendered, entity)
