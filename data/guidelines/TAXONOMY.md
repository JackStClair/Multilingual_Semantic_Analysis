# Nuance taxonomy (v1)

Closed label set for span annotation and BIO tagging. Prefer the **most specific** type that still fits; use `untranslatable_lexical` when a dense cultural package does not fit elsewhere.

| Type | Definition | EN→DE examples | EN→JA examples |
|------|------------|----------------|----------------|
| `political_social` | Public, civic, or political framing that shifts collectivity, nation, or rights discourse | freedom/liberty ↔ Freiheit; homeland ↔ Heimat | freedom ↔ 自由; public self presentation |
| `self_other` | Individual vs relational self; interdependence models | guilt/shame frames; belonging | 甘え (amae); 和 (wa); 本音/建前 |
| `affect_atmosphere` | Mood, atmosphere, or ambient social feeling | Gemütlichkeit; Sehnsucht | 空気; 懐かしい; 雰囲気 |
| `politeness_stance` | Stance, honorification, or address that English flattens | Siezen/Duzen lexicalized as formality | 敬語 load; お疲れ様; 遠慮 |
| `institutional_legal` | Legal, bureaucratic, or institutional concepts with different scope | Bildung; Beruf | 義務教育 framing; 会社 as social world |
| `untranslatable_lexical` | Compact lexical packages with weak one-word equivalents | Schadenfreude*; Weltschmerz | 木漏れ日; 生きがい |

\*Loanwords in English may still be annotated when the English gloss (“joy at misfortune”) strips German compound framing.

## Non-labels (do not annotate)

- Ordinary idioms with no cultural asymmetry (“kick the bucket”)
- Pure grammar / agreement
- Named entities without cultural-frame shift
- Generic sentiment words (“happy”, “sad”) unless tied to a documented frame mismatch
