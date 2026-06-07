# Context Distillation (2025)

## Overview

Context distillation is a method for internalizing a long context block (such as a system prompt, few-shot exemplars, or a RAG-retrieved document) directly into a model's weights. Once internalized, the model behaves on future inferences as if the context were present, even when it is omitted from the prompt. This reduces inference token costs and latency for static or slowly-changing contexts.

## Literature

The original baseline was established by Askell et al. (2021) in the Anthropic assistant alignment paper. They proposed a simple KL divergence loss between a teacher (which sees the context) and a student (which does not):
`Loss = KL(π_teacher(· | context, prompt) || π_student(· | prompt))`

However, subsequent work found that pure logit-matching on the generated response often fails to preserve factual information when the context is long or complex. The student learns the "style" of being context-conditioned but forgets the exact facts.

Recent 2024–2025 papers address this fact-forgetting problem:

1. **Deep Context Distillation with Plug-n-Play Modules (Caccia et al., Mar 2025)**
   This approach matches the hidden states of the teacher and student, not just the final logits. By applying an MSE loss on the hidden states of the last N transformer layers over the prompt tokens, the student is forced to internally represent the prompt exactly as the teacher does when conditioned on the context.

2. **Efficient Knowledge Injection in LLMs via Self-Distillation (OpenReview 2025)**
   This work highlights the need for explicit fact verification. By selectively weighting distillation examples based on whether the student successfully retained the injected knowledge, the distillation process becomes much more sample-efficient.

3. **On-Policy Context Distillation (OPCD)**
   Instead of forcing the student to match the teacher's outputs on a fixed dataset, the student generates its own rollouts (on-policy) and the teacher scores them. This reverse-KL approach avoids forcing the student to model low-probability teacher sequences.

## Implementation in Lile

The `ccd` (context-conditioned distillation) objective in Lile implements a hybrid approach based on the 2025 literature:
- **Logit KL**: A baseline `KL(π_teacher || π_student)` loss over the probe span.
- **Hidden-state matching (DCD)**: An optional `match_hidden` MSE loss on the last N transformer layers.
- **Fact Verification**: A custom `fact_verifier` that measures token-level F1 retention of ground-truth facts. This verifier drives a per-sample weight multiplier during training (upweighting probes where the student failed to retain the fact).

This objective is deployed as a standard LoRA module inside Lile's multi-objective training loop.
