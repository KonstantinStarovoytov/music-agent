---
name: spec-sync
description: Spec-driven workflow rules for this repo. Use before implementing any feature or changing agent behavior — spec.md is the source of truth.
user-invocable: false
---

# Spec-driven workflow

1. Before implementing anything, read `spec.md`. If the feature is not specified there, add/update the spec section first and show it to the user, then implement.
2. Node contracts (input/output Pydantic models) in code must match the contracts table in spec.md. If they diverge, the spec wins unless the user says otherwise.
3. When behavior changes, update spec.md in the same change set.
4. Deterministic logic (Camelot wheel neighbors, BPM tolerance, energy curve) lives in plain Python with unit tests — never inside prompts.
