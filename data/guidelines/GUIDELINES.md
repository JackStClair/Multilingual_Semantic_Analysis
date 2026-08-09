# Annotation guidelines

## Unit of annotation

One JSONL row = **one culturally asymmetric span** in an English source sentence (or short paragraph), for a specific `lang_pair` (`en-de` or `en-ja`).

Mark the **minimal English trigger** that carries the asymmetry (usually a word or short phrase). Do not mark the whole sentence unless the asymmetry is truly sentence-level (rare in v1).

## Decision procedure

1. Is there a plausible denotational translation into DE/JA?
2. Does a standard translation **erase or distort** a culturally documented frame?
3. Can you cite evidence (paper, reference grammar, cultural lexicon, reputable style/usage note)?
4. If yes to all: annotate with taxonomy type, frames, explanation, evidence.

## Positive examples

**EN→DE — homeland / Heimat**  
English *homeland* often denotes origin country; German *Heimat* layers emotional belonging, locality, and historically contested political load. Annotate `homeland` as `political_social` or `affect_atmosphere` depending on context; prefer `political_social` in civic discourse.

**EN→JA — dependence / 甘え**  
English *dependence* is often pathologized; Japanese *amae* (Doi) names an expected relational indulgence. Annotate `dependence` / `rely on` when the intended target frame is amae-like intimacy (`self_other`).

## Negative examples

- “It’s raining cats and dogs” — idiom, not cultural-frame asymmetry between EN and DE/JA for this task.
- Marking *the* or tense morphology.
- Annotating every instance of *please* without honorific collapse into 敬語.

## Frames

- `en_frame`: short gloss of the English cultural/default reading  
- `target_frame`: short gloss of the DE or JA reading that is lost or shifted  

## Confidence

- `high`: clear citation + prototypical context  
- `medium`: solid but context-sensitive  
- `low`: borderline; prefer exclude from train if unsure  

## LLM assistance policy

LLMs may propose candidate sentences or draft explanations. **Every retained row requires human acceptance and an evidence citation.** Hallucinated citations are reject criteria.
