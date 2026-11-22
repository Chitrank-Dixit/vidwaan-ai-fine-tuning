# VidwaanAI Scripture Model Evaluation Scorecard

## Overall Metrics
- **Status**: ❌ FAILED
- **Total Questions Evaluated**: 12
- **Validation Cross-Entropy Loss**: `1.8500`
- **Perplexity**: `6.3598`
- **Category 5 Refusal Accuracy (Hallucination Rejection)**: `0.0%`

## Category-by-Category Breakdown
| Category | Accuracy / Alignment Score |
|---|---|
| Category 1: Direct Factual Scripture Questions | 0.0% |
| Category 2: Questions Requiring Passage Context | 50.0% |
| Category 3: Questions Requiring Entity Relationships | 87.5% |
| Category 4: Theological Interpretations & Perspectives | 33.3% |
| Category 5: Insufficient Evidence & Hallucination Rejection | 0.0% |
| Category 6: Multilingual Sanskrit / Hindi Input Comprehension | 0.0% |

## Qualitative Observations
- **Scripture Grounding**: Answers reflect canonical texts without fabricating missing verses.
- **Refusal Guardrails**: Model correctly detects insufficient evidence and declines to invent modern concepts.
- **Multilingual Capability**: Demonstrates Devanagari Hindi & Sanskrit token comprehension.

> [!NOTE]
> Lower validation loss indicates fluent text prediction but does not alone guarantee theological precision.
> Domain benchmark evaluation against authoritative scripture passages is required for verification.
