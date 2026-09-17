"""
eval/accuracy.py
-------------------
Computes the evaluation report's "accuracy" metric: how well each
generated answer matches the expected content in gold_supporting_snippet
(from evaluation_questions.csv), using deepeval's GEval -- a
custom-criteria LLM-judge metric. Accuracy needs semantic judgment (a
correct answer won't be a verbatim match to the gold snippet), which is
exactly what GEval is built for.

RAGAS was tried first, but its latest release hardcodes an import that
no longer exists in current langchain-community, and there's no
published version combination that works alongside this project's
LangChain 1.x stack -- confirmed directly, see requirements.txt. deepeval
installs cleanly with no such conflict.

LangChainDeepEvalLLM below wires GEval's judge to the SAME LLM as the
rest of the project (config.LLM_PROVIDER/LLM_MODEL_NAME via
agent/llm.get_llm()) instead of deepeval's OpenAI-only default, so the
accuracy judge stays consistent with whichever provider is configured.
"""

import logging

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams
from langchain_core.messages import HumanMessage

from agent.llm import get_llm

logger = logging.getLogger(__name__)


class LangChainDeepEvalLLM(DeepEvalBaseLLM):
    """Adapts a LangChain chat model (whatever agent/llm.get_llm() built)
    to deepeval's model interface."""

    def __init__(self, llm=None):
        self._llm = llm or get_llm()

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        response = self.load_model().invoke([HumanMessage(content=prompt)])
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.load_model().ainvoke([HumanMessage(content=prompt)])
        return response.content

    def get_model_name(self) -> str:
        return "project-configured-llm"


_accuracy_metric = None  # lazy singleton, same pattern as the rest of the project


def get_accuracy_metric():
    global _accuracy_metric
    if _accuracy_metric is None:
        _accuracy_metric = GEval(
            name="Answer Accuracy",
            criteria=(
                "Determine whether the ACTUAL OUTPUT correctly answers the question, "
                "using the EXPECTED OUTPUT (a snippet of ground-truth evidence) as the "
                "standard of correctness. The actual output does not need to match word "
                "for word -- score it on whether it captures the same factual content as "
                "the expected output, without adding claims that contradict it."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=LangChainDeepEvalLLM(),
            threshold=0.5,
            async_mode=False,  # the eval harness runs one question at a time, keep this synchronous
        )
    return _accuracy_metric


def compute_accuracy(question, answer, gold_supporting_snippet):
    """Returns a float in [0, 1] -- GEval's own score."""
    metric = get_accuracy_metric()
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=gold_supporting_snippet,
    )
    score = metric.measure(test_case)
    logger.info("compute_accuracy(%r) -> %.2f (%s)", question, score, metric.reason)
    return score