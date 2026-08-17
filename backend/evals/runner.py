"""
Run the vision agent against the labelled cases and report what happened.

    python -m evals.runner                  # every case
    python -m evals.runner --case clean-total low-res-total
    python -m evals.runner --no-baseline    # skip the /vision/analyze comparison

Deliberately NOT part of the pytest suite. It needs a live API key, it
spends provider quota on every run, and its results are measurements rather
than assertions - a number that moved is information, not a build failure.
The graders it uses are unit-tested separately (tests/test_eval_grading.py),
so the scoring can be trusted without spending anything to check it.

THE BASELINE IS THE POINT

Each case also runs through VisionService.analyze_image directly - one call
to the vision model, which is what POST /vision/analyze does and what this
agent has to justify itself against. Accuracy in isolation says nothing:
the question stage 1 asked was whether deciding how to read an image beats
always doing the same thing, and only a side-by-side answers it.

WHAT THE NUMBERS ARE WORTH

The images are drawn, not photographed (see tests/image_builder.py), so
recognition performs better here than on real input. Read the results as an
upper bound and a regression signal, not as a prediction of production
accuracy.
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agents.vision_agent import VisionAgent
from ai.llm_service import get_llm_service
from ai.vision_service import get_vision_service
from app.core.config import settings
from evals.cases import CASES, EvalCase
from evals.grading import is_refusal, tool_choice_ok, tools_used, value_accuracy

REPORT_DIR = Path(__file__).parent / "reports"
PNG = "image/png"


@dataclass
class CaseResult:
    id: str
    passed: bool
    reason: str
    answer: str
    tools: list[str] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    unverified: list[str] = field(default_factory=list)
    stopped_because: str = ""
    # The single-vision-call comparison, when it was run.
    baseline_passed: bool | None = None
    baseline_answer: str = ""
    baseline_seconds: float = 0.0
    # True when the baseline call itself failed (a rate limit, a timeout).
    # Such a case is excluded from the comparison rather than counted
    # against the baseline.
    baseline_error: bool = False


def _build_agent() -> VisionAgent:
    from app.services.rag_service import build_retriever

    return VisionAgent(
        llm_service=get_llm_service(),
        vision_service=get_vision_service(),
        retriever=build_retriever(),
        max_steps=settings.vision_agent_max_steps,
        search_top_k=settings.vision_agent_search_top_k,
        temperature=settings.agent_temperature,
    )


def _grade(case: EvalCase, answer: str, steps) -> tuple[bool, str]:
    """Apply the case's expectations. Returns `(passed, reason)`."""
    if case.expect_refusal:
        if is_refusal(answer):
            return True, "declined, as it should"
        return False, "stated something instead of declining"

    hit, total = value_accuracy(answer, case.expect)
    if hit < total:
        missing = [
            f"{key}={value}"
            for key, value in case.expect.items()
            if not value_accuracy(answer, {key: value})[0]
        ]
        return False, f"missing {', '.join(missing)}"

    ok, reason = tool_choice_ok(steps, case.require_tools, case.forbid_tools)
    if not ok:
        # The facts were right but the route was wrong. Reported as a
        # failure on purpose: reaching the answer by running everything is
        # the behaviour this agent exists to avoid.
        return False, f"right answer, wrong route - {reason}"

    return True, "ok"


def run_case(agent: VisionAgent, case: EvalCase, baseline: bool) -> CaseResult:
    built = case.build()

    started = time.perf_counter()
    result = agent.run(case.question, image_bytes=built.png, mime_type=PNG)
    elapsed = time.perf_counter() - started

    passed, reason = _grade(case, result.answer, result.steps)

    outcome = CaseResult(
        id=case.id,
        passed=passed,
        reason=reason,
        answer=result.answer,
        tools=tools_used(result.steps),
        steps=len(result.steps),
        seconds=round(elapsed, 2),
        unverified=result.unverified_values,
        stopped_because=result.stopped_because,
    )

    if baseline:
        started = time.perf_counter()
        try:
            baseline_answer = get_vision_service().analyze_image(
                image_bytes=built.png,
                mime_type=PNG,
                question=case.question,
                temperature=settings.agent_temperature,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not graded
            # Left ungraded rather than scored as a wrong answer. The first
            # live run hit a 429 here and recorded it as a baseline failure,
            # which flattered the agent by a whole case - a rate limit says
            # nothing about whether one vision call can answer the question.
            outcome.baseline_answer = f"(baseline call failed: {exc})"
            outcome.baseline_error = True
            outcome.baseline_seconds = round(time.perf_counter() - started, 2)
            return outcome

        outcome.baseline_seconds = round(time.perf_counter() - started, 2)
        outcome.baseline_answer = baseline_answer
        # Graded on facts only. The baseline has no trace, so the
        # tool-choice half of the criterion does not apply to it.
        if case.expect_refusal:
            outcome.baseline_passed = is_refusal(baseline_answer)
        else:
            hit, total = value_accuracy(baseline_answer, case.expect)
            outcome.baseline_passed = hit == total

    return outcome


def summarise(results: list[CaseResult], baseline: bool) -> dict:
    passed = [r for r in results if r.passed]
    latencies = sorted(r.seconds for r in results)

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round(fraction * (len(values) - 1))))
        return values[index]

    summary = {
        "cases": len(results),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(results), 3) if results else 0.0,
        "median_steps": statistics.median([r.steps for r in results]) if results else 0,
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "runs_with_unverified_values": sum(1 for r in results if r.unverified),
        "runs_hitting_step_limit": sum(
            1 for r in results if r.stopped_because != "finished"
        ),
    }

    if baseline:
        # Only cases the baseline actually answered. A rate-limited call is
        # missing data, not a wrong answer, and the agent's own pass rate on
        # the same subset is what it should be compared against.
        graded = [r for r in results if r.baseline_passed is not None]
        summary["baseline_cases"] = len(graded)
        summary["baseline_errors"] = sum(1 for r in results if r.baseline_error)
        summary["baseline_passed"] = sum(1 for r in graded if r.baseline_passed)
        summary["baseline_pass_rate"] = (
            round(sum(1 for r in graded if r.baseline_passed) / len(graded), 3)
            if graded
            else 0.0
        )
        summary["agent_pass_rate_on_baseline_subset"] = (
            round(sum(1 for r in graded if r.passed) / len(graded), 3)
            if graded
            else 0.0
        )
        summary["baseline_latency_p50"] = percentile(
            sorted(r.baseline_seconds for r in graded), 0.50
        )

    return summary


def _print_report(results: list[CaseResult], summary: dict, baseline: bool) -> None:
    width = max(len(r.id) for r in results) + 2
    print()
    header = f"{'case'.ljust(width)}{'agent':>7}{'secs':>7}{'steps':>6}"
    if baseline:
        header += f"{'base':>7}"
    print(header)
    print("-" * len(header))

    for result in results:
        line = (
            f"{result.id.ljust(width)}"
            f"{('PASS' if result.passed else 'FAIL'):>7}"
            f"{result.seconds:>7.1f}"
            f"{result.steps:>6}"
        )
        if baseline:
            if result.baseline_error:
                line += f"{'err':>7}"
            elif result.baseline_passed is not None:
                line += f"{('PASS' if result.baseline_passed else 'FAIL'):>7}"
        print(line)
        if not result.passed:
            print(f"{' ' * width}  ↳ {result.reason}")
        if result.unverified:
            print(f"{' ' * width}  ↳ unverified: {', '.join(result.unverified)}")

    print()
    print(f"agent    {summary['passed']}/{summary['cases']} "
          f"({summary['pass_rate']:.0%})   "
          f"p50 {summary['latency_p50']:.1f}s   "
          f"p95 {summary['latency_p95']:.1f}s   "
          f"median {summary['median_steps']:.0f} steps")
    if baseline:
        comparable = summary["baseline_cases"]
        print(f"baseline {summary['baseline_passed']}/{comparable} "
              f"({summary['baseline_pass_rate']:.0%})   "
              f"p50 {summary['baseline_latency_p50']:.1f}s"
              + (f"   ({summary['baseline_errors']} call(s) failed, excluded)"
                 if summary["baseline_errors"] else ""))
        print(f"  \u2192 on those {comparable} cases the agent scored "
              f"{summary['agent_pass_rate_on_baseline_subset']:.0%}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="*", help="case ids to run (default: all)")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the single-vision-call comparison",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="where to write the JSON report"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help=(
            "seconds to pause between cases. The Groq free tier meters "
            "tokens per minute, and a run that trips it turns a measurement "
            "into a measurement of the rate limiter."
        ),
    )
    args = parser.parse_args(argv)

    if not settings.groq_api_key:
        print(
            "GROQ_API_KEY is not set. This runner makes live provider calls; "
            "add the key to backend/.env before running it.",
            file=sys.stderr,
        )
        return 2

    selected = CASES
    if args.case:
        wanted = set(args.case)
        selected = [case for case in CASES if case.id in wanted]
        unknown = wanted - {case.id for case in CASES}
        if unknown:
            print(f"unknown case ids: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2

    baseline = not args.no_baseline
    agent = _build_agent()

    results: list[CaseResult] = []
    for index, case in enumerate(selected, start=1):
        if index > 1 and args.delay:
            time.sleep(args.delay)
        print(f"[{index}/{len(selected)}] {case.id} …", flush=True)
        results.append(run_case(agent, case, baseline))

    summary = summarise(results, baseline)
    _print_report(results, summary, baseline)

    # Written so runs are comparable over time - the "iterate and improve"
    # half of the stage. Without a persisted report you can tell whether
    # today's numbers are good, but not whether they are better.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.out or (REPORT_DIR / f"{stamp}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "text_model": settings.groq_model,
                "vision_model": settings.groq_vision_model,
                "temperature": settings.agent_temperature,
                "max_steps": settings.vision_agent_max_steps,
                "summary": summary,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )
    print(f"report: {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
