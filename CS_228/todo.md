# Project TODO: Strengthening Bot-of-Thought (BoT) Implementation

This list consolidates improvements needed based on the expert critique and the comparison with Kedar's ReAct implementation.

## 🟥 High Priority: Core BoT Functional Improvements
- [x] **Implement Thought Instantiation:** Process the template with local facts (coordinates, distances, obstacle names) so the model receives a concrete situational plan.
- [x] **Enable Dynamic Buffer Updates:** Implement logic to update `usage_count` and `success_rate` in `ThoughtTemplate` after every episode.
- [x] **Implement Buffer Learning:** Replace low-performing templates with new "distilled" thoughts from successful high-reward episodes.

## 🟨 Medium Priority: Robustness & Scripting Fixes
- [x] **Proper Problem Distillation:** Implement `ProblemDistiller.distill` to return structured summaries (e.g., `agent_pos`, `goal_pos`, `dist_to_lava`) instead of raw text.
- [x] **Robust Success Detection:** Transition from `reward > 0` to a coordinate-based check (`agent_pos == goal_pos`) to prevent false positives/negatives.

## ⬛ Cancelled / Deferred
- [x] **Anchored Action Parsing:** (Cancelled: Not needed).
- [x] **Upgrade Retrieval:** (Cancelled: Current if/else works well).
- [x] **Manual Prompt Construction:** (Cancelled: `apply_chat_template` is working well).
- [x] **Expand History:** (Cancelled: Not considering for this implementation).
- [x] **Dynamic Buffer Sizing:** (Cancelled: Low priority). (Overkill for minigrid)
- [x] **Multi-Step Instantiation:** (Cancelled: Low priority).
- [x] **Ablation Study Support:** (Cancelled: Low priority).
- [x] **Thought-Action Format:** (Cancelled: Not considering).

## ✅ Completed Tasks
- [x] Replay Success Gallery (for visual verification).
- [x] Basic Observation Wrapper (Textual coordinates).
- [x] Initial Meta-Buffer implementation.
- [x] **vLLM Integration:** Evaluate moving from local HuggingFace `auto_map` to a vLLM server (like Kedar's) for faster inference speeds (2-3x speedup).