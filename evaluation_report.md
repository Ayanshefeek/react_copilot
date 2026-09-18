# Evaluation Report: LLM-only Baseline vs ReAct + Tools

| Mode | N | Grounded Precision | Accuracy |
|------|---|---------------------|----------|
| llm | 20 | 0.00 | 0.57 |
| react | 20 | 0.97 | 0.85 |

## Example Rows

**Q01** (llm): What does 'faithfulness' mean in LLM evaluation?
- Answer: In LLM evaluation, **faithfulness** usually means **how well the model’s output is grounded in the provided source information or context**.

More concretely, a faithful answer:
- **stays consistent with the given documents/prompt**
- **does not hallucinate unsupported facts**
- **does not invent de
- Citations: (none)
- Grounded precision: 0.0, Accuracy: 0.9

**Q01** (react): What does 'faithfulness' mean in LLM evaluation?
- Answer: In LLM evaluation, “faithfulness” means the model’s answer is supported by the retrieved evidence; it is also referred to as “groundedness.” [policy_eval_playbook.pdf:1] "Faithfulness means the answer is supported by retrieved evidence." [22_ai_evals.md] "faithfulness (groundedness)"
- Citations: [policy_eval_playbook.pdf:1] "Faithfulness means the answer is supported by retrieved evidence."; [22_ai_evals.md] "faithfulness (groundedness)"
- Grounded precision: 1.0, Accuracy: 1.0

**Q02** (llm): Name two defenses against prompt injection.
- Answer: Two common defenses against prompt injection are:

1. **Strict instruction hierarchy / separation of roles** — keep system and developer instructions clearly separated from untrusted user content, and tell the model to treat user-provided text as data, not instructions.

2. **Input sanitization and 
- Citations: (none)
- Grounded precision: 0.0, Accuracy: 0.2

**Q02** (react): Name two defenses against prompt injection.
- Answer: Two defenses against prompt injection are instruction hierarchy and strict tool schemas [07_prompt_injection.md] "Defenses include instruction hierarchy, strict tool schemas, and content filtering."
- Citations: [07_prompt_injection.md] "Defenses include instruction hierarchy, strict tool schemas, and content filtering."
- Grounded precision: 1.0, Accuracy: 0.2

**Q03** (llm): Why do agent systems use a max-steps limit?
- Answer: Agent systems use a max-steps limit mainly to prevent uncontrolled loops and to keep the run bounded.

Common reasons:
- **Avoid infinite or runaway behavior**: the agent may keep planning, retrying, or calling tools without ever finishing.
- **Control cost and latency**: each step can trigger model
- Citations: (none)
- Grounded precision: 0.0, Accuracy: 0.8

**Q03** (react): Why do agent systems use a max-steps limit?
- Answer: Agent systems use a max-steps limit to prevent infinite loops in execution. [policy_eval_playbook.pdf:1] "A max-steps limit prevents infinite loops in agent execution."
- Citations: [policy_eval_playbook.pdf:1] "A max-steps limit prevents infinite loops in agent execution."
- Grounded precision: 1.0, Accuracy: 1.0
