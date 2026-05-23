---
name: ultrafeedback-expert
description: >
  Tools and knowledge for exploring the UltraFeedback LLM preference dataset.
  Activate when: user asks about LLM preferences, model comparisons, or
  instruction quality in the UltraFeedback dataset.
---

# UltraFeedback Expert

## When to Use This Skill

**Trigger patterns:**
- "UltraFeedback" or "preference data" or "chosen vs rejected"
- "Which model is preferred" or "model comparison"
- "Find similar instructions" or "semantic search"

## Available Tools

| Tool | Type | What it does |
|------|------|--------------|
| `main.default.lookup_source_info` | UC SQL | Returns row count and sample for a source |
| `main.default.analyze_instruction` | UC Python | Analyzes instruction complexity |
| `main.default.compare_models` | UC SQL | Compares two models by chosen vs rejected counts |
| `main.default.get_model_win_rate` | UC SQL | Returns a model's win rate in the dataset |
| `main.default.classify_instruction_topic` | UC Python | Classifies instruction topic by keywords |
| `main.default.search_similar_instructions` | UC SQL | Keyword-based search over dataset instructions |
| You.com MCP (Databricks) | External MCP | Live web search via Databricks proxy |

## Procedures

### Answering "What sources are in the dataset?"
1. Call `lookup_source_info` for each known source.
2. Summarize counts and sample instructions.

### Finding similar instructions
1. Call `search_similar_instructions` with the user's text.
2. Return the top 3-5 matches with their sources and similarity scores.

## Gotchas
- Embeddings table (`main.default.ultrafeedback_embeddings`) must exist with pre-computed vectors.
- Column names with hyphens (e.g., `chosen-model`) need backtick escaping.