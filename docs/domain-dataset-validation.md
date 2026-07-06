# Domain dataset validation report (extension E6)

Validation record for `src/bielik_hallu/dataset/candidates_domains.py`:
three entity domains (cities, writers, musicians) × three conditions
(KNOWN / UNKNOWN_REAL / FABRICATED) × 42 entities (42 per condition —
deliberate; see Adams 1979). Built 2026-07-04 as the round-2 extension E6
(cross-domain generalization, Ferrando-style), incorporating the M2 remedy
from `paper/review_round1.md`: the Bielik v3 family has **two** tokenizers
(1.5B/4.5B vs Minitron-7B/11B), so fabricated names are token-matched to the
KNOWN lists under **both**.

Token counts everywhere below use the pipeline's definition
(`tokenization.token_length`: `len(tokenizer(entity, add_special_tokens=False))`
on the bare entity string).

## 1. Method

### KNOWN
Hand-picked, unquestionably famous Polish entities. For **cities**, the
composition is constrained by the M2 lesson: 19 of the most famous Polish
cities tokenize to a *single* token under the 1.5B/4.5B tokenizer, while an
invented toponym can never be 1 token, which puts a hard floor on the
achievable K-vs-F token AUROC. The final list therefore keeps 9 iconic
one-token giants (Warszawa, Kraków, Wrocław, Poznań, Gdańsk, Szczecin, Łódź,
Katowice, Gdynia) and fills the remaining slots with equally famous
multi-token cities/towns (Zamość, Przemyśl, Gniezno, Wadowice, Wieliczka,
Ciechocinek, ...). With 19 one-token entries the floor would have been
≈ 0.65; with 9 it drops to ≈ 0.54.

### UNKNOWN_REAL
Sourced from verifiable registries via the live Polish-Wikipedia API
(2026-07-04):

- **Cities**: village articles (TERYT-backed — pl.wiki has an article for
  every TERYT locality) harvested by walking the category trees
  `Kategoria:Wsie w województwie {małopolskim, opolskim, podkarpackim}` and
  `Kategoria:Wsie w powiecie słupskim` down to gmina level and sampling
  villages. These are the only voivodeship-level village category trees on
  pl.wiki, so the sample skews to southern Poland (documented limitation).
  Titles with parenthesised disambiguators were excluded.
- **Writers**: sampled from stub articles (≤ 3.5 kB page length, two-word
  names, no disambiguators) in `Kategoria:Polscy poeci ludowi`,
  `Kategoria:Polscy poeci XX wieku`, `Kategoria:Polscy prozaicy XX wieku`.
- **Musicians**: same stub filter over `Kategoria:Polscy muzycy ludowi`,
  `Polscy skrzypkowie`, `Polscy organiści`, `Polscy muzycy folkowi`,
  `Polscy kompozytorzy muzyki poważnej`.

Every UNKNOWN_REAL entry (42 × 3 = 126) therefore has its existence verified
by construction against the live registry (article exists, category and page
size checked at harvest time). Recognisable-despite-stub names surfaced by
the harvest were rejected by hand (see §3). Additionally, the first 15
entries per domain were spot-checked with DuckDuckGo exact-phrase search
(§4).

### FABRICATED
Generated as morphologically valid mutations, athletes-list style:

- **People**: light surname mutations of the domain's KNOWN names
  (suffix/vowel swaps: Sienkiewicz→Sienkowski, Osiecka→Osieńska,
  Penderecki→Penderak) plus invented names for token-length variety.
  Hand-authored pools of 109 (writers) / 102 (musicians) candidates.
- **Cities**: invented toponyms from productive Polish toponymic morphology
  (stems brzoz-/kalin-/sikor-/wilg-/... × suffixes -ice, -ów, -owo, -iny,
  -niki, -sko, -ęcin, plus compounds), pool of ~700 candidates over several
  iterations, with hand-review of junction phonotactics (rejecting e.g.
  "Grabnnik", "Brzozsko", "Piaskowowice").

**Non-existence screening** (every pool candidate, live web, 2026-07-04):

1. pl.wikipedia exact-title lookup, redirect-aware → must be missing.
   For cities this is a strong registry check: every real Polish locality
   has a TERYT-backed article.
2. pl.wikipedia search: people = exact-phrase full-text search (0 hits
   required); cities = `intitle:"name"` search (0 hits required; full-text
   phrase search stem-matches common nouns and over-flags toponyms).
3. DuckDuckGo exact-phrase web check of **all 126 final names** (lite +
   html endpoints). Success criterion: no result mentioning the exact name
   as a real notable person/place. This stage caught one real locality that
   Wikipedia screening missed (Borowniki, a Wołyń village) — see §3.

**Token matching**: from each screened pool, the final 42 names were chosen
by local-search optimisation (random swap, 100k iterations, 5 seeds)
minimising `max(|AUROC_small − 0.5|, |AUROC_large − 0.5|)` where AUROC_x is
the K-vs-F separability of entity token count under tokenizer x, with a
diversity penalty capping names per stem/surname family (≤ 3 for cities,
≤ 2 for people).

## 2. Token-count statistics (final lists)

Tokenizers: **small** = `speakleash/Bielik-1.5B-v3.0-Instruct` (shared with
4.5B), **large** = `speakleash/Bielik-11B-v3.0-Instruct` (shared with
Minitron-7B).

### cities

| tokenizer | condition | mean | median | min | max |
|---|---|---|---|---|---|
| small | KNOWN | 2.12 | 2 | 1 | 5 |
| small | UNKNOWN_REAL | 3.36 | 3 | 2 | 6 |
| small | FABRICATED | 2.45 | 2 | 2 | 3 |
| large | KNOWN | 3.90 | 3 | 2 | 8 |
| large | UNKNOWN_REAL | 4.74 | 5 | 2 | 9 |
| large | FABRICATED | 3.71 | 4 | 2 | 6 |

### writers

| tokenizer | condition | mean | median | min | max |
|---|---|---|---|---|---|
| small | KNOWN | 3.76 | 4 | 2 | 8 |
| small | UNKNOWN_REAL | 4.02 | 4 | 3 | 7 |
| small | FABRICATED | 3.62 | 4 | 3 | 6 |
| large | KNOWN | 6.45 | 6 | 3 | 11 |
| large | UNKNOWN_REAL | 6.21 | 6 | 3 | 11 |
| large | FABRICATED | 6.36 | 6 | 3 | 9 |

### musicians

| tokenizer | condition | mean | median | min | max |
|---|---|---|---|---|---|
| small | KNOWN | 3.81 | 4 | 2 | 6 |
| small | UNKNOWN_REAL | 3.67 | 4 | 3 | 6 |
| small | FABRICATED | 3.83 | 4 | 3 | 7 |
| large | KNOWN | 6.43 | 6 | 2 | 10 |
| large | UNKNOWN_REAL | 6.24 | 6 | 3 | 10 |
| large | FABRICATED | 6.45 | 6 | 4 | 10 |

### Token-count-only AUROC (lexical baseline)

| domain | contrast | small (1.5B/4.5B) | large (11B/Minitron) |
|---|---|---|---|
| cities | **K vs F** | **0.637** | **0.494** |
| cities | K vs UR | 0.816 | 0.663 |
| cities | UR vs F | 0.251 | 0.293 |
| writers | **K vs F** | **0.500** | **0.500** |
| writers | K vs UR | 0.606 | 0.459 |
| writers | UR vs F | 0.370 | 0.552 |
| musicians | **K vs F** | **0.509** | **0.505** |
| musicians | K vs UR | 0.441 | 0.467 |
| musicians | UR vs F | 0.581 | 0.539 |

All three K-vs-F AUROCs are below the 0.65 target under both tokenizers
(athletes round-1 baseline: 0.68 / 0.613, review issue M2). Writers and
musicians are matched to chance (0.50). Cities saturate at 0.637 under the
small tokenizer: the 9 one-token KNOWN giants cannot be matched by any
invented toponym (minimum 2 tokens), and the clean 2-token fabricated supply
is finite; the optimiser reached the floor of the screened pool. Only K vs F
was optimised; UNKNOWN_REAL names are natural real names and keep their
natural token statistics (the K-vs-UR / UR-vs-F columns are reported for
transparency — note UR-vs-F is *below* 0.5 where fabricated names are
shorter than village names).

## 3. Replacements and rejections log

### Screened out automatically (pool stage)

- **Cities**: 102 of the ~700 generated toponyms existed as pl.wiki titles
  (e.g. Brzozowice, Kalinowo, Jawornik, Malinówka, Topolany) and 56 more had
  full-text/intitle hits (e.g. Wierzbnów, Jarzębiny, Malinin, Pstrągów) —
  all dropped. Later iterations dropped further title collisions
  (Szczygłów, Drozdowiec, Jarząbkowo, Sójkowo, Remizów, Przepiórów,
  Gniewomirowice, Kaliniec, Wroniec, Sarnice, Gołębice, Lisice, Żurawiec,
  Sikorzyce, ... — full lists in the pipeline artifacts).
- **Writers**: "Jan Kochanowicz" — pl.wiki title exists (actor); dropped
  from the pool.
- **Musicians**: "Waldemar Kilarski", "Konrad Krawczykowski",
  "Irena Santorska" — exact-phrase hits on pl.wiki (real private persons in
  article text); dropped from the pool.

### Hand-rejected (morphology / confusability / notability)

- Mechanical stem+suffix combos with broken junction phonotactics
  (Grabnnik, Dąbrno, Brzozsko, Sarnno, Wilgno, Piaskowowice, Olsziny,
  Smolnany, Szczyglowa, Głogowówka, ...) — excluded and re-optimised.
- "Kalininy/Kalinin" (Soviet-name collision), "Szczygłów"-adjacent spelling
  variants, and two-word forms built on *real* village bases
  ("Kunice Małe", "Czaplice Małe", ...) — excluded.
- Pun-like occupational surnames in the musicians pool (Fletniarska,
  Smyczkowska, Gęślicki, ...) — replaced with neutral morphology before
  screening.
- UNKNOWN_REAL harvest rejections (too recognisable despite stub size or
  not Polish-language writers): Filip Jaślar, Izydor Lotto, Karol Kątski,
  Jerzy Fitelberg, Bolesław Karpiel-Bułecka, Mateusz Gliński, Aleksander
  Jarzębski, Józef Deszczyński, Wiera Badalska, Robert Jaworski, Eddie
  Courts, Walenty Krząszcz, Dawid Fryszman, Mejlech Chmielnicki (the last
  two write in Hebrew/Yiddish → swapped for Ryszard Binkowski and
  Włodzimierz Chełmicki).
- KNOWN-cities rejections for token-floor reasons (§1): Lublin, Białystok,
  Toruń, Częstochowa, Bydgoszcz, Rzeszów, Kielce, Olsztyn, Radom, Zabrze
  (all 1-token) — replaced by famous multi-token towns.

### Caught by the search-engine stage (final-list checks)

- **"Borowniki"** (cities) — real populated place (historical Wołyń
  village; mapcarta/wolyn.ovh hits) despite passing the pl.wiki screen.
  **Replaced.**
- **"Bocianice"** (cities) — ambiguous web hits (commercial pages).
  **Replaced** out of caution.
- **"Agata Chylewska"** (musicians) — real published chemist (University
  of Gdańsk co-authorships surfaced by search). **Replaced** by
  "Fryderyk Szopenowicz" (web-checked clean).
- **"Kinga Jaczewska"** (musicians) — real artist with a personal website
  and public profile. **Replaced** by "Anna Germańska" (web-checked: only
  a private social-media profile, no notable entity).

Kept after review (documented, not replaced): "Edyta Górniewicz" (appears
only in a regional professional registry) and "Piotr Dygoń" (appears only
in a voivodeship expert list) — non-notable private-person matches of the
kind that plausible Polish name mutations unavoidably produce (the athletes
FABRICATED list has the same property).

After the replacements each affected selection was re-optimised
(musicians post-swap K-vs-F token AUROC: 0.509 small / 0.505 large); the
final 42 per domain all pass every screening stage.

## 4. Web-verification coverage

- **FABRICATED, all 126 names**: pl.wiki exact-title (redirect-aware) miss
  + pl.wiki search 0 hits + a general search-engine exact-phrase check for
  every name. Engine coverage: cities 42/42 and writers 42/42 via
  DuckDuckGo (lite/html endpoints, waves with cool-downs); musicians 42/42
  via a second search engine after DuckDuckGo began hard rate-limiting
  (HTTP 403). Zero exact-name hits (or only private-person registry
  matches) interpreted as "no notable real entity with this exact name".
  Findings and replacements are logged in §3.
- **UNKNOWN_REAL, all 126 names**: existence verified per-name via live
  pl.wiki registry queries at harvest time (village articles are
  TERYT-backed; people from stub articles in writer/musician categories).
- **UNKNOWN_REAL search-engine spot checks (first 15 per domain)**:
  exact-phrase searches expected to return hits confirming real existence
  — results in §5.

## 5. UNKNOWN_REAL spot-check results (search engine, 15 per domain)

- **cities: 15/15 confirmed.** All fifteen surfaced as real villages with
  GUS/TERYT-backed statistics pages, geoportal entries, or Wikipedia
  articles (e.g. Zembrzyce and Śleszowice in gmina Zembrzyce; Orelec in
  Lesko County; Dobrucowa in gmina Tarnowiec; Wańkowa with its historic
  oil-mining centre; Dobcza near Sieniawa).
- **writers: 14/15 confirmed** by name in search results (e.g. Sebastian
  Lesiczka — folk poet and long-time wójt of Jeżowe; Tobiasz Stullich —
  Masurian folk poet, 1841–1908; Wincenty Motas — peasant poet from
  Grębów, 1860–1918; Frycz Olszewski — one of the first Masovian folk
  poets). "Ignacy Ślusarczyk" did not surface in engine results and
  remains verified by its live pl.wiki stub article only.
- **musicians: 10/15 confirmed** by name in search results (e.g. Antoni
  Cichecki — folk violinist, 1934–2018, Kapela Braci Cicheckich;
  Katarzyna Bąkowska — violin professor in Bydgoszcz; Roland Orlik, Jacek
  Niwelt, Małgorzata Górnisiewicz, Ludwik Holcman — listed Polish
  violinists; Mateusz Dziubiński, Karol Mikuszewski, Marian Bujak — listed
  folk musicians). The remaining five (Jarosław Śliwa, Bronisław Zaremba,
  Danuta Głowacka, Kazimierz Olechowski, Józef Bryjka) were diluted out of
  the batched engine queries and remain verified by their live pl.wiki
  stub articles.

## 6. Reproducibility

Pipeline scripts (scratchpad artifacts, not part of the repo):
pool generation → Wikipedia screening (exact title, phrase/intitle search)
→ tokenizer profiling (both HF tokenizers, offline cache) → AUROC-targeted
local-search selection → DuckDuckGo verification waves. All Wikipedia and
DDG queries executed live on 2026-07-04. The final lists are frozen as
literals in `candidates_domains.py`; `tests/test_candidates_domains.py`
locks counts (42/42/42 × 3), uniqueness (within domain, across domains, and
against the athletes lists), prompt templates, and name hygiene.
