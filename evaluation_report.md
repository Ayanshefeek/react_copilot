# Evaluation Report: LLM-only Baseline vs ReAct + Tools

| Mode | N | Grounded Precision | Accuracy |
|------|---|---------------------|----------|
| llm | 20 | 0.00 | 0.47 |
| react | 20 | 0.85 | 0.71 |

## Example Rows

**Q01** (llm): What does 'faithfulness' mean in LLM evaluation?
- Answer: In the context of evaluating large language models (LLMs), 'faithfulness' refers to the degree to which the generated outputs accurately reflect the information or claims present in the input data or source material. It assesses whether the model's responses are truthful, reliable, and consistent wi
- Citations: (none)
- Grounded precision: 0.0, Accuracy: 0.4

**Q01** (react): What does 'faithfulness' mean in LLM evaluation?
- Answer: In the context of LLM (Large Language Model) evaluation, "faithfulness" refers to the degree to which an answer is supported by retrieved evidence. It is often associated with the concept of "groundedness," indicating that the information provided by the model should be verifiable and based on relia
- Citations: [policy_eval_playbook.pdf:1] "Faithfulness means the answer is supported by retrieved evidence."; [22_ai_evals.md] "faithfulness (groundedness)."
- Grounded precision: 1.0, Accuracy: 0.7

**Q02** (llm): Name two defenses against prompt injection.
- Answer: Two defenses against prompt injection are:

1. **Input Validation and Sanitization**: Ensuring that any user input is thoroughly validated and sanitized can help prevent malicious prompts from being executed. This involves checking for unexpected characters or patterns that could alter the intended 
- Citations: (none)
- Grounded precision: 0.0, Accuracy: 0.5

**Q02** (react): Name two defenses against prompt injection.
- Answer: Two defenses against prompt injection are:

1. **Instruction Hierarchy** - This involves establishing a clear hierarchy of instructions to ensure that critical system commands are prioritized over user inputs.
2. **Strict Tool Schemas** - Implementing strict schemas for tools can help validate and c
- Citations: [07_prompt_injection_001] "Prompt injection attempts to override system instructions or exfiltrate hidden data."; [07_prompt_injection_002] "Defenses include instruction hierarchy, strict tool schemas, and content filtering."
- Grounded precision: 1.0, Accuracy: 0.4

**Q03** (llm): Why do agent systems use a max-steps limit?
- Answer: Agent systems use a max-steps limit to prevent infinite loops and ensure that the system can terminate or make progress within a reasonable timeframe. This limit helps manage computational resources effectively, allowing for predictable performance and enabling the system to handle tasks in a timely
- Citations: (none)
- Grounded precision: 0.0, Accuracy: 0.8

**Q03** (react): Why do agent systems use a max-steps limit?
- Answer: Agent systems use a max-steps limit primarily to prevent infinite loops during execution. This safeguard ensures that the system does not get stuck in a cycle of actions without making progress. [policy_eval_playbook.pdf:1] "A max-steps limit prevents infinite loops in agent execution."
- Citations: [policy_eval_playbook.pdf:1] "A max-steps limit prevents infinite loops in agent execution."
- Grounded precision: 1.0, Accuracy: 0.9
