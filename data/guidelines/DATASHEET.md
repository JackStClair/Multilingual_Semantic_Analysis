# Datasheet: Cultural Asymmetry Spans (EN–DE / EN–JA)

## Motivation

Machine translation systems optimize lexical/semantic equivalence and fluency. They under-specify **cultural framing** differences that matter to bilingual readers. This dataset supports span detection and explanation of such asymmetries.

## Composition

- Language pairs: `en-de`, `en-ja` (English source spans)
- Instance type: short English contexts (1–3 sentences) with character-offset span annotations
- Labels: closed taxonomy in `TAXONOMY.md`
- Fields: see schema in repository README / annotation JSONL

## Collection process

1. Seed lexicons compiled from linguistics and cultural studies references (see lexicon `sources` fields).
2. Mini-contexts authored or adapted to host seed items naturally.
3. Human annotation of spans, frames, explanations, and evidence.
4. External expansion (v0.2): Koharu24 MyUWD DE/JA sheets + INK-USC untranslatability corpus (JA); see `data/external/SOURCES.md`. Imports are `confidence: medium`.
5. Train/val/test split by source sentence (random) for v1 trainability.

## Recommended uses

- Token-level cultural-nuance detection
- Explanation retrieval for translation UX
- Error analysis of MT systems on culturally loaded vocabulary

## Non-recommended uses

- Treating explanations as ground-truth anthropology without reading cited sources
- High-stakes legal/medical translation without expert review
- Claiming native-speaker consensus from a small curated set

## Ethical considerations

Cultural categories can essentialize. Annotations describe **documented discourse frames**, not immutable national psychologies. Prefer hedging in UI copy. Japanese and German items include historically sensitive political terms (e.g. Heimat); explanations must note contestation where relevant.

## Maintenance

Prototype dataset for the `nuance-mt` project. Expand via the same guidelines; bump version in filenames when taxonomy changes.
