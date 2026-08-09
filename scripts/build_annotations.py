#!/usr/bin/env python3
"""Build train/val/test JSONL and explanation store from seed lexicons."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEX_DIR = ROOT / "data" / "seed_lexicon"
OUT_DIR = ROOT / "data" / "annotations"
EXP_PATH = ROOT / "data" / "explanations" / "store.jsonl"

# Multiple natural mini-contexts per trigger family (English hosts).
CONTEXTS_DE = {
    "homeland": [
        "After years abroad, he still longed for his homeland in a way that maps never captured.",
        "Politicians invoked the homeland to mobilize voters who felt culturally adrift.",
        "She returned to her homeland not for opportunity, but for a belonging she lacked overseas.",
    ],
    "home soil": [
        "Standing again on home soil, he felt a pull stronger than nostalgia for a house.",
    ],
    "native home": [
        "The novel treats the native home as a moral landscape, not a GPS coordinate.",
    ],
    "cozy": [
        "The café was cozy, yet the word felt too small for the shared ease in the room.",
        "They wanted a cozy evening, meaning unhurried togetherness rather than soft pillows alone.",
    ],
    "coziness": [
        "Tourism ads sell coziness, but locals hear a social warmth English under-describes.",
    ],
    "snug atmosphere": [
        "Critics praised the snug atmosphere of the inn as something closer to communal ease.",
    ],
    "education": [
        "Their debate about education was really about forming a person, not passing exams.",
        "She valued education as cultivation of judgment, not only job skills.",
    ],
    "cultivation": [
        "The curriculum still spoke of cultivation in a formative, almost moral sense.",
    ],
    "self-cultivation": [
        "Self-cultivation mattered more to him than credentials on a résumé.",
    ],
    "freedom": [
        "They argued about freedom: one meant non-interference, the other a civic good.",
        "In the pamphlet, freedom sounded less like private choice and more like public emancipation.",
    ],
    "liberty": [
        "The speech praised liberty in an individualist key that the German audience heard differently.",
    ],
    "guilt": [
        "He spoke of guilt as if it were also a debt he still owed the community.",
        "The memoir refuses cheap guilt; it digs into obligation with a heavier grain.",
    ],
    "fault": [
        "Assigning fault felt narrower than the Schuld he kept naming in German.",
    ],
    "blame": [
        "Blame in English pointed to a person; his German drafts kept sliding toward owed reckoning.",
    ],
    "longing": [
        "Her longing for the mountains had a bittersweet intensity tourists flatten into wanderlust.",
    ],
    "yearning": [
        "Romantic yearning fills the poem—more Sehnsucht than a simple want.",
    ],
    "deep longing": [
        "A deep longing runs through the letters, almost too heavy for 'I miss you'.",
    ],
    "world-weariness": [
        "His world-weariness was philosophical, a mismatch between ideals and the age.",
    ],
    "world weariness": [
        "Critics called it world weariness; he meant a Romantic diagnosis of the times.",
    ],
    "existential weariness": [
        "The essay's existential weariness is closer to Weltschmerz than burnout.",
    ],
    "joy at misfortune": [
        "She admitted a flicker of joy at misfortune—and hated how English moralized it.",
    ],
    "pleasure at others' pain": [
        "The study treats pleasure at others' pain as a nameable emotion, not only a sin.",
    ],
    "gloating": [
        "What English calls gloating, the German compound names without the sermon.",
    ],
    "use first names": [
        "When they agreed to use first names, the German side heard a duzen threshold.",
    ],
    "be on familiar terms": [
        "To be on familiar terms at work still required an explicit address shift.",
    ],
    "switch to informal address": [
        "A clumsy attempt to switch to informal address created weeks of awkwardness.",
    ],
    "formal address": [
        "He kept formal address carefully, treating Siezen as stance rather than stiffness.",
    ],
    "use formal you": [
        "Learners struggle when to use formal you because English hides the binary.",
    ],
    "keep formal distance": [
        "She chose to keep formal distance even after years of collaboration.",
    ],
    "profession": [
        "He spoke of his profession with a vocational gravity English 'job' lacks.",
    ],
    "vocation": [
        "Calling it a vocation risked sounding sacred; Beruf sat in between.",
    ],
    "calling": [
        "The old sense of calling still ghosts modern talk of Beruf.",
    ],
    "anxiety": [
        "Her anxiety shaded into a broader dread that German speakers might call Angst.",
    ],
    "dread": [
        "A free-floating dread fills the novel—more Angst than fear of a object.",
    ],
    "existential anxiety": [
        "The seminar on existential anxiety kept sliding into German philosophical Angst.",
    ],
    "wanderlust": [
        "Wanderlust sounded cheerful; what he felt was an ache for elsewhere.",
    ],
    "itch to travel": [
        "The itch to travel was less excitement than Fernweh.",
    ],
    "desire to be elsewhere": [
        "A desire to be elsewhere haunted winters more than summers.",
    ],
    "homesickness": [
        "Homesickness only partly translates the Heimweh tied to Heimat talk.",
    ],
    "homesick": [
        "He was homesick in a way that implicated belonging, not only his childhood bedroom.",
    ],
    "order": [
        "Calls for order in the neighborhood carried a moral tone beyond neat streets.",
    ],
    "orderliness": [
        "Their praise of orderliness sounded like a social virtue, not aesthetics.",
    ],
    "keeping order": [
        "Keeping order was framed as a shared duty, almost proverbial.",
    ],
    "duty": [
        "Duty in his essay still echoed a Kantian Pflicht rather than a chore list.",
    ],
    "obligation": [
        "Obligation thickened into Pflicht whenever the argument turned moral.",
    ],
    "mind": [
        "Translating Geist as mind cut away the spirit/intellect continuum.",
    ],
    "spirit": [
        "Spirit overshot; Geist was doing more philosophical work than ghosts.",
    ],
    "intellect": [
        "Intellect alone missed how Geist organizes the human sciences naming.",
    ],
    "think along": [
        "Managers asked staff to think along—anticipate the shared task unprompted.",
    ],
    "anticipate needs": [
        "To anticipate needs was praised as mitdenken, not merely customer service.",
    ],
    "think ahead with others": [
        "Good teammates think ahead with others without waiting for tickets.",
    ],
    "tact": [
        "Tact was too small; he wanted Fingerspitzengefühl for the negotiation.",
    ],
    "intuition": [
        "Her intuition in politics was fingertip-fine situational judgment.",
    ],
    "delicate intuition": [
        "Diplomacy needed delicate intuition more than loud principle.",
    ],
    "security": [
        "Children need security as sheltered belonging—Geborgenheit—not only locks.",
    ],
    "feeling safe": [
        "Feeling safe in her grandmother's kitchen was emotional shelter.",
    ],
    "safe belonging": [
        "The policy talk of safe belonging strained to match Geborgenheit.",
    ],
    "mischievous": [
        "His mischievous smile was wry and knowing, almost verschmitzt.",
    ],
    "knowing smile": [
        "A knowing smile crossed her face—playful without childishness.",
    ],
    "impish": [
        "The impish aside softens the critique without undoing it.",
    ],
    "lead culture": [
        "Debates over a lead culture revive the charged Leitkultur keyword.",
    ],
    "core culture": [
        "Appeals to a core culture rarely stay culturally neutral in German media.",
    ],
    "dominant culture": [
        "Dominant culture is a flat gloss for Leitkultur's political charge.",
    ],
    "coming to terms with the past": [
        "Coming to terms with the past names a specifically German postwar project.",
    ],
    "working through the past": [
        "Working through the past is closer to Vergangenheitsbewältigung than generic apology.",
    ],
    "middle-class": [
        "Middle-class respectability in the novel tracks bürgerlich norms.",
    ],
    "bourgeois": [
        "Bourgeois here means civic-cultural respectability, not only income.",
    ],
    "respectable middle-class": [
        "A respectable middle-class ethos shapes the family's silences.",
    ],
    "secondhand embarrassment": [
        "Secondhand embarrassment has a neat German name: Fremdscham.",
    ],
    "vicarious embarrassment": [
        "Vicarious embarrassment flooded her during the presentation.",
    ],
}

CONTEXTS_JA = {
    "dependence": [
        "Her dependence on her mentor was not weakness; it was a culturally legible intimacy.",
        "English called it dependence; he heard amae—an appeal for indulgent care.",
    ],
    "dependent": [
        "Being dependent in that household was expected at certain ages, not shameful.",
    ],
    "rely on": [
        "Children learn when they may rely on adults without being labeled needy.",
    ],
    "lean on": [
        "She could lean on her sister in a way that assumed indulgent acceptance.",
    ],
    "harmony": [
        "They prized harmony enough to soften disagreement before it became public.",
        "Group harmony shaped what could be said aloud in the meeting.",
    ],
    "group harmony": [
        "Preserving group harmony mattered more than a blunt factual win.",
    ],
    "keep the peace": [
        "He chose to keep the peace, reading wa as a duty rather than niceness.",
    ],
    "true feelings": [
        "True feelings stayed private while the public stance did institutional work.",
    ],
    "public face": [
        "Her public face was careful; colleagues did not confuse it with deceit.",
    ],
    "real feelings versus public stance": [
        "The novel maps real feelings versus public stance without moral panic.",
    ],
    "reading the room": [
        "He was good at reading the room before speaking, prioritizing unspoken norms.",
    ],
    "read the room": [
        "Failure to read the room hurt him more than a wrong spreadsheet.",
    ],
    "atmosphere": [
        "The atmosphere demanded restraint; someone had to name the kuuki.",
    ],
    "air of the room": [
        "The air of the room shifted when the director entered.",
    ],
    "nostalgic": [
        "The smell made her unexpectedly nostalgic—warm recognition more than melancholy.",
    ],
    "nostalgia": [
        "Nostalgia only partly fits the natsukashii rush of a childhood snack.",
    ],
    "fondly missed": [
        "A song she fondly missed returned as bodily familiarity.",
    ],
    "good work": [
        "At quitting time, good work functioned as ritual acknowledgment of shared labor.",
    ],
    "thanks for your hard work": [
        "Thanks for your hard work is thinner than the reciprocal fatigue in otsukaresama.",
    ],
    "you must be tired": [
        "Saying you must be tired can be care or formula; context decides.",
    ],
    "restraint": [
        "Her restraint was enryo—an ethic of non-imposition, not mere shyness.",
    ],
    "hold back": [
        "Guests hold back before taking the last piece as a default stance.",
    ],
    "don't impose": [
        "Children are taught don't impose long before they learn English politeness kits.",
    ],
    "hesitate to impose": [
        "He would hesitate to impose even when help was offered twice.",
    ],
    "serious": [
        "Calling her serious missed that majime praised reliability, not gloom.",
    ],
    "earnest": [
        "An earnest employee can be socially valued beyond humorlessness.",
    ],
    "diligent": [
        "Diligent barely covers the character term majime implies.",
    ],
    "endurance": [
        "Endurance was framed as virtue—gaman—rather than temporary patience.",
    ],
    "put up with it": [
        "They told him to put up with it, and the phrase carried moral weight.",
    ],
    "grin and bear it": [
        "Grin and bear it approximates gaman but sounds more flippant.",
    ],
    "patience": [
        "Patience understates gaman's characterological endurance.",
    ],
    "life purpose": [
        "Life purpose posters oversell what ikigai means in ordinary roles.",
    ],
    "reason for living": [
        "A quiet reason for living can be daily care work, not a grand passion.",
    ],
    "what gets you up": [
        "What gets you up in the morning is a thin popular gloss of ikigai.",
    ],
    "sunlight through leaves": [
        "Photographers chase sunlight through leaves as if English lacked a single word.",
    ],
    "dappled light in trees": [
        "Dappled light in trees is a phrase where Japanese offers komorebi.",
    ],
    "understated": [
        "The pottery was understated in a shibui key—restrained, almost astringent.",
    ],
    "austere beauty": [
        "Austere beauty still undersells shibui's taste metaphor.",
    ],
    "subtle taste": [
        "Subtle taste in dress can signal shibui refinement.",
    ],
    "thanks for the food": [
        "Thanks for the food is variable; itadakimasu is a fixed humble receiving.",
    ],
    "let's eat": [
        "Let's eat misses the ritual humble verb before the meal.",
    ],
    "I humbly receive": [
        "I humbly receive is calqued, but points at itadakimasu's stance.",
    ],
    "what a waste": [
        "What a waste only partly captures mottainai's respect for unused value.",
    ],
    "too good to waste": [
        "The leftover tofu was too good to waste—mottainai in a kitchen key.",
    ],
    "wasteful": [
        "Calling it wasteful moralizes consumption; mottainai adds reverence for potential.",
    ],
    "senior": [
        "A senior classmate remained a sempai with duties years after graduation.",
    ],
    "junior": [
        "As a junior hire, he inherited kōhai obligations automatically.",
    ],
    "mentor junior hierarchy": [
        "The mentor junior hierarchy was structural, not optional networking.",
    ],
    "continuous improvement": [
        "Continuous improvement as slogan is thinner than shop-floor kaizen histories.",
    ],
    "incremental improvement": [
        "Incremental improvement sounded managerial; veterans heard kaizen practice.",
    ],
    "hospitality": [
        "Hospitality ads borrowed omotenashi while flattening anticipatory care.",
    ],
    "wholehearted hospitality": [
        "Wholehearted hospitality still misses omotenashi's no-tip anticipatory ethic.",
    ],
    "it can't be helped": [
        "It can't be helped is the everyday shrug of shōganai.",
    ],
    "nothing you can do": [
        "Nothing you can do closes the topic without drama.",
    ],
    "it is what it is": [
        "It is what it is approximates shōganai, with less ritual weight.",
    ],
    "do your best": [
        "Coaches say do your best; ganbaru saturates encouragement more deeply.",
    ],
    "hang in there": [
        "Hang in there is cousin to ganbatte, not a perfect twin.",
    ],
    "keep at it": [
        "Keep at it names perseverance; ganbaru names a social script.",
    ],
    "in-group": [
        "In-group boundaries shaped language choice more than English 'us' talk.",
    ],
    "out-group": [
        "Out-group guests received different honorific care.",
    ],
    "insiders and outsiders": [
        "Mapping insiders and outsiders is basic to uchi/soto analysis.",
    ],
    "lay the groundwork": [
        "Managers lay the groundwork quietly—nemawashi before the formal vote.",
    ],
    "consensus building beforehand": [
        "Consensus building beforehand is expected, not underhanded.",
    ],
    "pre-meeting alignment": [
        "Pre-meeting alignment prevents public confrontation.",
    ],
    "rustic beauty": [
        "Rustic beauty gestures at wabi-sabi without the full aesthetic complex.",
    ],
    "beauty of imperfection": [
        "The beauty of imperfection is a popular English slogan for wabi-sabi.",
    ],
    "weathered beauty": [
        "Weathered beauty in the tea bowl invited a wabi-sabi reading.",
    ],
    "freedom": [
        "They translated freedom as jiyū, carrying modern Japanese political history with it.",
    ],
    "liberty": [
        "Liberty talk from Atlantic sources sits awkwardly atop jiyū's local genealogy.",
    ],
    "fateful connection": [
        "They called it a fateful connection—en rather than networking.",
    ],
    "destiny bond": [
        "A destiny bond sounds mystical; en is ordinary and deep at once.",
    ],
    "deep connection": [
        "A deep connection may be en when framed as relational fate.",
    ],
    "cute": [
        "Cute understates how far kawaii organizes aesthetics and evaluation.",
    ],
    "adorable": [
        "Adorable is narrower than the social category kawaii.",
    ],
}


def _find_span(text: str, trigger: str) -> tuple[int, int]:
    idx = text.lower().find(trigger.lower())
    if idx < 0:
        raise ValueError(f"trigger {trigger!r} not in {text!r}")
    return idx, idx + len(trigger)


def load_lexicon(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows() -> list[dict]:
    rows: list[dict] = []
    contexts = {"en-de": CONTEXTS_DE, "en-ja": CONTEXTS_JA}
    for lang_pair, filename in [("en-de", "en-de.json"), ("en-ja", "en-ja.json")]:
        lex = load_lexicon(LEX_DIR / filename)
        for item in lex["items"]:
            for trigger in item["en_triggers"]:
                for ctx in contexts[lang_pair].get(trigger, []):
                    start, end = _find_span(ctx, trigger)
                    # Minimal reference translation stub (human-editable later)
                    translation = _stub_translation(ctx, lang_pair, item)
                    rid = f"{item['id']}:{trigger}:{abs(hash(ctx)) % 10_000_000:07d}"
                    rows.append(
                        {
                            "id": rid,
                            "lang_pair": lang_pair,
                            "source_text": ctx,
                            "translation": translation,
                            "src_char_start": start,
                            "src_char_end": end,
                            "src_span": ctx[start:end],
                            "tgt_span": item["tgt_span"],
                            "nuance_type": item["nuance_type"],
                            "en_frame": item["en_frame"],
                            "target_frame": item["target_frame"],
                            "explanation": item["explanation"],
                            "evidence": item["evidence"],
                            "annotator": "seed-v0.1",
                            "confidence": "high",
                            "lexicon_id": item["id"],
                        }
                    )
    return rows


def _stub_translation(ctx: str, lang_pair: str, item: dict) -> str:
    # Placeholder parallel line for schema completeness; NLLB used at inference.
    if lang_pair == "en-de":
        return f"[de-ref] {ctx} ⟶ focus: {item['tgt_span']}"
    return f"[ja-ref] {ctx} ⟶ focus: {item['tgt_span']}"


def split_rows(rows: list[dict], seed: int = 13):
    # Split by lexicon_id to reduce leakage of the same concept across splits
    by_lex: dict[str, list[dict]] = {}
    for r in rows:
        by_lex.setdefault(r["lexicon_id"], []).append(r)
    ids = sorted(by_lex)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(0.15 * n))
    n_val = max(1, int(0.15 * n))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train, val, test = [], [], []
    for lid, items in by_lex.items():
        if lid in test_ids:
            test.extend(items)
        elif lid in val_ids:
            val.extend(items)
        else:
            train.extend(items)
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_explanations(rows: list[dict]) -> None:
    seen: set[tuple[str, str, str]] = set()
    EXP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXP_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            key = (r["lang_pair"], r["src_span"].lower(), r["nuance_type"])
            if key in seen:
                continue
            seen.add(key)
            f.write(
                json.dumps(
                    {
                        "lemma": r["src_span"].lower(),
                        "lang_pair": r["lang_pair"],
                        "nuance_type": r["nuance_type"],
                        "en_frame": r["en_frame"],
                        "target_frame": r["target_frame"],
                        "explanation": r["explanation"],
                        "evidence": r["evidence"],
                        "tgt_span": r["tgt_span"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    rows = build_rows()
    train, val, test = split_rows(rows)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)
    write_explanations(rows)
    print(
        f"Wrote {len(train)} train / {len(val)} val / {len(test)} test "
        f"(total {len(rows)}) → {OUT_DIR}"
    )
    print(f"Explanation store → {EXP_PATH}")


if __name__ == "__main__":
    main()
