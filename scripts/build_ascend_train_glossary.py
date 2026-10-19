#!/usr/bin/env python3
"""Build a frozen, train-only glossary and an independent distractor control.

This module's rule and reservoirs were fixed before reading the training export.
It deliberately imports no project runner, models, or scoring outcomes. Input is
the transcript-only JSONL export with exactly id, reference, and language fields.
No corpus-dependent choices, synonyms, manual replacements, or thresholds exist.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


RULE_VERSION = "ascend-train-content-df-v2"
MIN_DOCUMENT_FREQUENCY = 3
UNIGRAM_COUNT = 20
BIGRAM_COUNT = 10
ASCII_CONTENT = re.compile(r"[a-z]{2,}\Z")
ANNOTATION_BOUNDARY = "__square_bracket_annotation__"

# Independent closed list of English grammatical/function words, contractions
# after punctuation splitting, and common conversational fillers. Never inferred
# from a corpus. Other content words are not manually added to this list.
STOP_WORDS = frozenset("""
a an the and or but if then than as at by for from in into of off on onto out over
per to up upon via with without about above across after against along among
around before behind below beneath beside between beyond down during except
inside near outside past since through throughout till toward towards under
underneath until within
i me my mine myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves who whom whose which what whatever whoever whichever
this that these those here there where when why how
am is are was were be been being do does did doing have has had having
can could may might must shall should will would ought need dare
not no nor neither none nothing nobody never cannot
all any both each either enough every everybody everyone everything few fewer
less little many more most much other others several some somebody someone
something such same own one ones another
very too so just only even ever also already again almost always away back
else especially quite rather really still well yet perhaps probably maybe
yes yeah yep yup no nope okay ok oh ah eh er erm uh uhm um hmm hm huh hey
like actually basically literally anyway anyways anyhow sorta kinda
s t d ll m re ve don doesn didn isn aren wasn weren hasn haven hadn won wouldn
couldn shouldn mustn ain let lets please thanks thank hello hi goodbye bye
""".split())

# Fixed independent scientific/geographic vocabulary reservoir. Its only
# corpus-dependent filtering is exclusion of all train-vocabulary collisions.
# Controls are assigned by length distance, never by recognition performance.
UNRELATED_SINGLE_RESERVOIR = tuple(sorted(set("""
tor erg ria fen col tarn fir yew elm ash oak bog scree talus loess esker kame
karst cirque arete dune mesa butte fjord atoll basalt gabbro diorite
dacite rhyolite schist gneiss phyllite quartz jasper chert shale slate marl
dolomite calcite gypsum halite olivine epidote augite albite anorthite
zeolite beryl garnet spinel rutile zircon apatite fluorite pyrite galena
sphalerite bornite azurite malachite chalcopyrite hematite goethite limonite
cuesta horst graben drumlin moraine outwash nunatak yardang inselberg
hoodoo doline polje uvala lapilli tephra pumice scoria obsidian ignimbrite
lahar fumarole solfatara batholith laccolith sill dike syncline anticline
monocline isostasy orogen craton terrane bathymetry hypsometry regolith
laterite saprolite duricrust caliche aridisol vertisol alfisol spodosol
podzol rendzina gleysol histosol andisol mollisol oxisol ultisol entisol
sepal tepal carpel anther stamen pistil stigma ovule sorus strobilus
rhizome corm tuber stolon xylem phloem cambium pith petiole stipule
bract cyme raceme panicle umbel catkin spadix achene samara caryopsis
mericarp drupe pome silique capsule nutlet cotyledon hypocotyl epicotyl
radicle coleoptile coleorhiza perianth gametophyte sporophyte archegonium
antheridium protonema rhizoid thallus frond soralia isidia mycobiont
photobiont diatom desmid rotifer tardigrade bryozoan crinoid brachiopod
trilobite ammonite belemnite graptolite stromatolite oncolite ooid
oolite peloid rudist conodont nautiloid ichnite coprolite palynology
""".split())))

UNRELATED_BIGRAM_RESERVOIR = tuple(sorted(set("""
abyssal plain|aeolian loess|alluvial fan|alpine tundra|andesitic tephra|
antithetic fault|aphanitic basalt|archean craton|axillary meristem|
barbed awn|basaltic dike|bathyal zone|benthic foraminifera|bilabiate corolla|
braided esker|calcareous ooze|calcic plagioclase|capillary fringe|
carboniferous cyclothem|chasmophytic fern|chloritic schist|ciliate margin|
colloidal silica|columnar jointing|conchoidal fracture|contact aureole|
cordate lamina|coriaceous bract|crenulate margin|crinoidal limestone|
cryogenic regolith|cuneate base|dendritic drainage|diagenetic cement|
distichous phyllotaxy|doleritic sill|dolomitic marl|drumlin swarm|
epeiric basin|epiclastic breccia|epigynous ovary|equant crystal|
esker ridge|exstipulate petiole|ferruginous duricrust|filiform stigma|
fluvial terrace|foliated gneiss|foraminiferal ooze|fumarolic vent|
fusiform achene|glabrous sepal|glaciofluvial outwash|granitic batholith|
graptolitic shale|gypsiferous mudstone|halophytic sedge|hanging syncline|
heterosporous lycophyte|hummocky moraine|hyaline tepal|hydric histosol|
hypabyssal intrusion|imbricate aestivation|incised meander|indusiate sorus|
intertidal mudflat|isoclinal fold|karstic doline|lacustrine diatomite|
lanceolate leaflet|lateritic saprolite|lenticular lens|lepidote trichome|
lithic arenite|loessal paleosol|mafic xenolith|marly chalk|
mesophytic herb|metamorphic aureole|microcrystalline chert|monoclinic gypsum|
mucronate apex|mycorrhizal rhizome|nepheline syenite|nodular flint|
obcordate petal|oolitic limestone|ophitic texture|orbicular stipule|
palynological assemblage|papillose epidermis|peltate scale|periglacial solifluction|
phreatic aquifer|phyllitic cleavage|pinnate venation|plutonic gabbro|
porphyritic rhyolite|proglacial varve|pyroclastic ignimbrite|racemose cyme|
recurved sepal|reticulate venation|rhombohedral calcite|rhizoidal filament|
roche moutonnee|saccate pollen|sagittate anther|schistose foliation|
scoriaceous lapilli|serrate leaflet|siliceous sinter|spheroidal weathering|
spicate panicle|staminate catkin|stipitate ovary|stromatolitic dolomite|
subglacial till|tabular xenocryst|talus apron|tectonic klippe|
tholeiitic basalt|tomentose bract|trilobitic coquina|tuffaceous siltstone|
umbellate inflorescence|undulose extinction|verticillate phyllotaxy|
vesicular pumice|xylitic lignite|zeugen ridge|zirconiferous sandstone
""".replace("\n", "").split("|"))))

RULE = {
    "version": RULE_VERSION,
    "selection_partition": "train only",
    "row_filter": "language equals mixed (exact match)",
    "transcript_sanitization": "after Unicode NFKC, remove all square-bracket annotation spans (nested spans included; an unmatched opening bracket extends to end of reference); retain a non-content token boundary for each outermost span; apply to both candidate extraction and all-train vocabulary",
    "normalization": "Unicode NFKC; lowercase; Han characters separate; letters and decimal digits in separate runs; punctuation separates",
    "content_token": "ASCII alphabetic token of at least 2 letters, absent from fixed stop_words",
    "bigram": "two consecutive normalized tokens that are both content tokens; do not bridge other tokens",
    "rank": ["document_frequency descending", "total_frequency descending", "term lexicographic ascending"],
    "minimum_document_frequency": MIN_DOCUMENT_FREQUENCY,
    "select_unigrams": UNIGRAM_COUNT,
    "select_bigrams": BIGRAM_COUNT,
    "insufficient_candidates": "fail; no fallback or threshold relaxation",
    "unrelated_reservoir": "fixed independent scientific/geographic vocabulary in extractor source",
    "unrelated_exclusion": "exclude every reservoir term with any normalized token observed in any train row; no tokens reused across selected controls",
    "unrelated_assignment": "for each correct term in selection order: choose unused same-token-count control by absolute total-letter-count distance, summed corresponding-token-length distance, maximum corresponding-token-length distance, then lexical order",
    "stop_words": sorted(STOP_WORDS),
    "no_outcome_adaptation": True,
}


def _is_han(char: str) -> bool:
    name = unicodedata.name(char, "")
    return name.startswith(("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-"))


def normalize_tokens(text: str) -> list[str]:
    """Scorer-compatible normalization, kept standalone to avoid dependencies."""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    pending: list[str] = []
    kind: str | None = None

    def flush() -> None:
        if pending:
            tokens.append("".join(pending))
            pending.clear()

    for char in text:
        if _is_han(char):
            flush()
            kind = None
            tokens.append(char)
        elif char.isdecimal():
            if kind != "digits":
                flush()
            pending.append(char)
            kind = "digits"
        elif char.isalpha():
            if kind != "word":
                flush()
            pending.append(char)
            kind = "word"
        elif unicodedata.category(char).startswith("M") and kind == "word":
            pending.append(char)
        else:
            flush()
            kind = None
    flush()
    return tokens


def is_content(token: str) -> bool:
    return ASCII_CONTENT.fullmatch(token) is not None and token not in STOP_WORDS


def transcript_tokens(text: str) -> tuple[list[str], int, bool]:
    """Discard bracketed annotations while preserving their adjacency barrier.

    Count outermost spans so nested annotation text is removed as one span.
    A malformed, unclosed opening bracket conservatively removes the remainder.
    Reservoir terms use normalize_tokens directly; this applies only to corpus
    transcripts and therefore cannot modify the independent control reservoir.
    """
    tokens: list[str] = []
    segment: list[str] = []
    depth = 0
    span_count = 0
    for char in unicodedata.normalize("NFKC", text):
        if char == "[":
            if depth == 0:
                tokens.extend(normalize_tokens("".join(segment)))
                segment.clear()
                tokens.append(ANNOTATION_BOUNDARY)
                span_count += 1
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif depth == 0:
            segment.append(char)
    tokens.extend(normalize_tokens("".join(segment)))
    return tokens, span_count, depth != 0


def read_train(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read exactly one caller-specified transcript-only training file."""
    raw = path.read_bytes()
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(row, dict) or set(row) != {"id", "reference", "language"}:
            raise ValueError(f"Line {line_number} must contain exactly id, reference, language")
        if any(not isinstance(row[key], str) for key in row):
            raise ValueError(f"Line {line_number} fields must be strings")
        if not row["id"] or row["id"] in seen_ids:
            raise ValueError(f"Line {line_number} has an empty or duplicate id")
        seen_ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Training export is empty")
    return rows, hashlib.sha256(raw).hexdigest()


def collect_candidates(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict], set[str], dict]:
    totals: dict[int, Counter] = {1: Counter(), 2: Counter()}
    documents: dict[int, dict[str, list[str]]] = {1: defaultdict(list), 2: defaultdict(list)}
    all_train_vocabulary: set[str] = set()
    sanitization = {
        "annotation_span_count": 0,
        "affected_row_count": 0,
        "unclosed_span_count": 0,
        "mixed_annotation_span_count": 0,
        "mixed_affected_row_count": 0,
        "mixed_unclosed_span_count": 0,
    }
    for row in rows:
        tokens, annotation_spans, unclosed = transcript_tokens(row["reference"])
        all_train_vocabulary.update(token for token in tokens if token != ANNOTATION_BOUNDARY)
        sanitization["annotation_span_count"] += annotation_spans
        sanitization["affected_row_count"] += int(annotation_spans != 0)
        sanitization["unclosed_span_count"] += int(unclosed)
        if row["language"] != "mixed":
            continue
        sanitization["mixed_annotation_span_count"] += annotation_spans
        sanitization["mixed_affected_row_count"] += int(annotation_spans != 0)
        sanitization["mixed_unclosed_span_count"] += int(unclosed)
        unigrams = [token for token in tokens if is_content(token)]
        bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])
                   if is_content(left) and is_content(right)]
        for size, terms in ((1, unigrams), (2, bigrams)):
            totals[size].update(terms)
            for term in set(terms):
                documents[size][term].append(row["id"])

    def ranked(size: int) -> list[dict]:
        candidates = [
            {"term": term, "token_count": size,
             "document_frequency": len(documents[size][term]),
             "total_frequency": frequency,
             "source_ids": sorted(documents[size][term]),
             "eligible": len(documents[size][term]) >= MIN_DOCUMENT_FREQUENCY}
            for term, frequency in totals[size].items()
        ]
        candidates.sort(key=lambda item: (-item["document_frequency"], -item["total_frequency"], item["term"]))
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank
        return candidates

    return ranked(1), ranked(2), all_train_vocabulary, sanitization


def select_correct(unigrams: list[dict], bigrams: list[dict]) -> list[dict]:
    selected = []
    for name, candidates, count in (("unigrams", unigrams, UNIGRAM_COUNT),
                                    ("bigrams", bigrams, BIGRAM_COUNT)):
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if len(eligible) < count:
            raise ValueError(f"Need {count} {name} with document frequency >= {MIN_DOCUMENT_FREQUENCY}; found {len(eligible)}. Frozen rule has no fallback.")
        selected.extend(eligible[:count])
    return selected


def select_unrelated(correct: list[dict], train_vocabulary: set[str]) -> tuple[list[str], list[dict], list[dict]]:
    reservoirs = {1: UNRELATED_SINGLE_RESERVOIR, 2: UNRELATED_BIGRAM_RESERVOIR}
    audit = []
    usable: dict[int, list[str]] = {1: [], 2: []}
    for token_count, reservoir in reservoirs.items():
        for term in reservoir:
            tokens = normalize_tokens(term)
            if len(tokens) != token_count or any(ASCII_CONTENT.fullmatch(token) is None for token in tokens):
                raise ValueError(f"Invalid fixed reservoir term: {term!r}")
            collisions = sorted(set(tokens) & train_vocabulary)
            audit.append({"term": term, "token_count": token_count,
                          "train_token_collisions": collisions, "eligible": not collisions})
            if not collisions:
                usable[token_count].append(term)

    selected: list[str] = []
    assignments: list[dict] = []
    used_tokens: set[str] = set()
    for candidate in correct:
        source = candidate["term"]
        lengths = list(map(len, source.split()))

        def distance(term: str) -> tuple:
            other = list(map(len, term.split()))
            per_token = [abs(left - right) for left, right in zip(lengths, other)]
            return (abs(sum(lengths) - sum(other)), sum(per_token), max(per_token), term)

        choices = [term for term in usable[len(lengths)]
                   if not (set(term.split()) & used_tokens)]
        if not choices:
            raise ValueError("Independent reservoir exhausted after train-vocabulary exclusion; no fallback")
        term = min(choices, key=distance)
        usable[len(lengths)].remove(term)
        used_tokens.update(term.split())
        selected.append(term)
        assignments.append({"correct_term": source, "unrelated_term": term,
                            "token_count": len(lengths),
                            "train_document_frequency": 0,
                            "train_total_frequency": 0,
                            "train_source_ids": [],
                            "correct_token_lengths": lengths,
                            "unrelated_token_lengths": list(map(len, term.split())),
                            "length_distance": list(distance(term)[:-1])})
    return selected, assignments, audit


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build(train_jsonl: Path, output_dir: Path) -> dict:
    train_jsonl = train_jsonl.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; refusing overwrite: {output_dir}")
    rows, source_sha = read_train(train_jsonl)
    unigrams, bigrams, vocabulary, sanitization = collect_candidates(rows)
    correct = select_correct(unigrams, bigrams)
    unrelated, assignments, reservoir_audit = select_unrelated(correct, vocabulary)
    correct_terms = [candidate["term"] for candidate in correct]
    payloads = {"correct.json": json_bytes(correct_terms), "unrelated.json": json_bytes(unrelated)}
    language_counts = dict(sorted(Counter(row["language"] for row in rows).items()))
    extraction = {
        "rule": RULE,
        "provenance": {
            "input_path": str(train_jsonl), "input_sha256": source_sha,
            "extractor_path": str(Path(__file__).resolve()),
            "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_row_count": len(rows), "language_row_counts": language_counts,
            "mixed_row_count": language_counts.get("mixed", 0),
            "all_train_vocabulary_count": len(vocabulary),
            "rule_sha256": hashlib.sha256(json_bytes(RULE)).hexdigest(),
            "reservoir_sha256": hashlib.sha256(json_bytes({1: UNRELATED_SINGLE_RESERVOIR, 2: UNRELATED_BIGRAM_RESERVOIR})).hexdigest(),
            "output_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
        },
        "term_counts": {"correct": len(correct_terms), "unrelated": len(unrelated),
                        "unigrams_per_glossary": UNIGRAM_COUNT, "bigrams_per_glossary": BIGRAM_COUNT},
        "transcript_sanitization_counts": sanitization,
        "selected_correct": correct,
        "unigram_candidates": unigrams,
        "bigram_candidates": bigrams,
        "unrelated_assignments": assignments,
        "unrelated_reservoir_audit": reservoir_audit,
    }
    payloads["extraction.json"] = json_bytes(extraction)
    # All validation/selection completes before creating the new destination.
    # mkdir(exist_ok=False) also guards against a concurrent directory creator.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(raw)
    return extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True,
                        help="transcript-only training JSONL: id, reference, language")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    try:
        result = build(args.train_jsonl, args.output_dir)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "input_sha256": result["provenance"]["input_sha256"],
                      "mixed_row_count": result["provenance"]["mixed_row_count"],
                      "term_counts": result["term_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
