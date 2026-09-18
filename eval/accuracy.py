"""
eval/accuracy.py
-------------------
Computes the evaluation report's "accuracy" metric via deepeval's
GEval, comparing each answer to its gold_supporting_snippet.

LangChainDeepEvalLLM wires GEval's judge to get_judge_llm() by default
(not the main agent's get_llm()) -- see config.JUDGE_LLM_PROVIDER to
run judging on a different/cheaper model than the ReAct agent uses.
"""

import logging

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams
from langchain_core.messages import HumanMessage

from agent.llm import get_judge_llm

logger = logging.getLogger(__name__)


class LangChainDeepEvalLLM(DeepEvalBaseLLM):
    """Adapts a LangChain chat model to deepeval's model interface."""

    def __init__(self, llm=None):
        self._llm = llm or get_judge_llm()

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        response = self.load_model().invoke([HumanMessage(content=prompt)])
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.load_model().ainvoke([HumanMessage(content=prompt)])
        return response.content

    def get_model_name(self) -> str:
        return "project-configured-judge-llm"


_accuracy_metric = None


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
            async_mode=False,
        )
    return _accuracy_metric


def compute_accuracy(question, answer, gold_supporting_snippet):
    metric = get_accuracy_metric()
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=gold_supporting_snippet,
    )
    score = metric.measure(test_case)
    logger.info("compute_accuracy(%r) -> %.2f (%s)", question, score, metric.reason)
    return score