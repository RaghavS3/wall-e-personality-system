# Wall-e Personality System

This repository is a public-safe bundle of the prompt, posture, memory, and evaluation code behind a personal-message assistant personality.

The core idea: personality is not one magic system prompt. It is a runtime made of a voice contract, deterministic posture flags, relationship memory, anti-slop rules, and behavioral evals.

## File Map

| File | What it demonstrates |
| --- | --- |
| `wall_e/server.py` | Main voice contract, memory contract, model-turn envelope, and memory consolidation pipeline. |
| `wall_e/interaction_boundary.py` | Deterministic identity, seriousness, humor, callback, and relationship-maturity routing. |
| `scripts/behavioral_eval/scenarios.json` | Behavior scenarios that turn personality into testable product requirements. |
| `scripts/behavioral_eval/rules.py` | Voice-quality lint rules for assistant-y phrasing, fake tool claims, forced relatability, prompt leakage, and weak refusals. |
| `scripts/behavioral_eval/scoring.py` | Scoring helpers that separate guardrail correctness from voice quality. |
| `scripts/tests/test_voice_contract.py` | Regression tests for the production voice contract. |
| `scripts/tests/test_identity_boundary.py` | Tests for identity and prompt-boundary behavior. |
| `scripts/tests/test_behavioral_eval.py` | Tests for the behavioral evaluation harness and rules. |
| `poke_prompt_baseline.py` | Earlier/simple Poke-style persona prompt, useful as the "before" example. |

## Article Framing

The assistant felt different because the system removes generic assistant behavior instead of merely asking the model to be more human.

Useful terms:

- **Voice contract:** the model-facing instruction layer that defines the assistant's style and hard avoid list.
- **Interaction posture:** structured flags computed before generation, such as serious mode, humor allowed, earned callback, and relationship maturity.
- **Anti-slop check:** a deletion pass for sentences that only validate, reassure, offer help, ask permission, or sound like customer support.
- **Relationship compression:** memory extraction that saves only durable facts, preferences, real callbacks, open loops, and meaningful emotional context.
- **Personality evals:** scenarios and lint rules that make taste measurable.

## Safety Notes

This bundle intentionally excludes local `.env` files, logs, SQLite databases, state files, caches, and scratch experiment output.

External model and messaging API access is disabled in this public bundle. The copied runtime keeps the voice, memory, posture, and eval architecture for study, but the Gemini, DeepSeek, Linq, and Twilio send paths raise before making network requests.
