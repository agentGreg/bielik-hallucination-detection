import re

import pytest
from bielik_hallu import config
from bielik_hallu.dataset import candidates as athletes
from bielik_hallu.dataset.candidates_domains import (
    DOMAINS,
    PROMPT_TEMPLATE_PEOPLE,
    PROMPT_TEMPLATE_PLACES,
)

EXPECTED_DOMAINS = {"cities", "writers", "musicians"}
# 42 entities per condition — deliberate; see Adams (1979).
N_PER_CONDITION = 42


def _all_names(domain):
    return (DOMAINS[domain]["KNOWN"] + DOMAINS[domain]["UNKNOWN_REAL"]
            + DOMAINS[domain]["FABRICATED"])


def test_domains_keys():
    assert set(DOMAINS.keys()) == EXPECTED_DOMAINS


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
def test_conditions_present(domain):
    assert set(config.CONDITIONS) <= set(DOMAINS[domain].keys())


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
@pytest.mark.parametrize("condition", ["KNOWN", "UNKNOWN_REAL", "FABRICATED"])
def test_counts(domain, condition):
    assert len(DOMAINS[domain][condition]) == N_PER_CONDITION


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
def test_no_duplicates_within_or_across_conditions(domain):
    names = _all_names(domain)
    assert len(names) == 3 * N_PER_CONDITION
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
def test_no_overlap_with_athletes_lists(domain):
    athlete_names = (set(athletes.KNOWN) | set(athletes.UNKNOWN_REAL)
                     | set(athletes.FABRICATED))
    assert not set(_all_names(domain)) & athlete_names


def test_no_overlap_across_domains():
    seen = {}
    for domain in EXPECTED_DOMAINS:
        for name in _all_names(domain):
            assert name not in seen, f"{name} in both {seen.get(name)} and {domain}"
            seen[name] = domain


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
def test_prompt_template_present_and_formats(domain):
    template = DOMAINS[domain]["prompt_template"]
    assert "{entity}" in template
    rendered = template.format(entity="X")
    assert rendered.endswith("Odpowiedz jednym zdaniem.")
    assert "X" in rendered


def test_people_domains_use_kim_jest():
    assert DOMAINS["writers"]["prompt_template"] == PROMPT_TEMPLATE_PEOPLE
    assert DOMAINS["musicians"]["prompt_template"] == PROMPT_TEMPLATE_PEOPLE
    assert PROMPT_TEMPLATE_PEOPLE.startswith("Kim jest ")
    # people template matches the athletes pipeline template
    assert PROMPT_TEMPLATE_PEOPLE == config.PROMPT_TEMPLATE


def test_cities_domain_uses_czym_jest():
    assert DOMAINS["cities"]["prompt_template"] == PROMPT_TEMPLATE_PLACES
    assert PROMPT_TEMPLATE_PLACES.startswith("Czym jest ")


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
def test_names_nonempty_and_clean(domain):
    for name in _all_names(domain):
        assert isinstance(name, str)
        assert name == name.strip()
        assert name
        assert "  " not in name


@pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
def test_names_title_cased_sensibly(domain):
    # every space- or hyphen-separated word starts with an uppercase letter
    for name in _all_names(domain):
        for word in re.split(r"[ \-]", name):
            assert word, f"empty word segment in {name!r}"
            assert word[0].isupper(), f"{name!r}: segment {word!r} not capitalized"
