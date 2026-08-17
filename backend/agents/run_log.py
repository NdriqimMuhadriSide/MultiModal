"""
One structured log line per agent run, and the parentage that links them.

WHY THIS EXISTS

Until delegation, an agent run was one loop and the trace it returned in the
HTTP response was the whole story. A supervisor changes that: a single
question now produces a *tree* of runs, and the questions worth asking about
it are the ones a per-request trace cannot answer.

    Which layer spent the steps - the supervisor, or one specialist it
    called four times?
    Is the p95 that moved yesterday the research agent, or the vision call
    inside it?
    How often does a delegated run come back having hit its budget rather
    than finishing?

None of those are visible in a response body, because each one is a question
about many runs rather than about this one. So every run - root or delegated -
emits a record, and every record carries the id of the run that caused it.

WHY LOGGING AND NOT A METRICS SERVICE

There is no queue, no collector and no time-series database in this project,
and adding one to answer these questions would be a larger decision than the
questions warrant. A structured line on the existing logger is greppable
today, and is the shape you would ship to a collector later without changing
any call site. If that day comes, `_emit` is the only function that has to
know.

HOW PARENTAGE IS TRACKED

A `contextvars.ContextVar`, not a parameter threaded through the loop.

Sub-agents are built once at dependency-injection time (see
app/services/agent_service.py) and invoked synchronously deep inside a tool's
`run`, so there is no call signature in between that could carry a parent id
without every layer agreeing to pass it along. A context variable is exactly
the tool for a value that is scoped to a dynamic extent rather than to an
argument list: the supervisor sets it for the duration of its run, every
sub-agent started inside that run reads it, and the token is reset on the way
out so a second root run in the same thread does not inherit the first one's
id.

Context is per-thread-copy under Starlette's threadpool, which is the correct
behaviour here: two concurrent requests each get their own tree.
"""
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# The run that is currently executing on this context, or None at the root.
# Read only by `record_run`, which is what makes a delegated run know who
# called it without anybody passing it down.
_current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)

# The root of the current tree. Carried separately from the parent so a whole
# tree can be pulled out of the logs with one grep, without having to walk
# parent links back up from the leaves.
_root_run_id: ContextVar[str | None] = ContextVar("root_run_id", default=None)


@dataclass
class RunRecord:
    """
    What one agent run did, as it will be logged.

    Mutable during the run: `steps`, `tools` and `stopped_because` are filled
    in by the loop as it goes, and `seconds` is stamped on the way out. The
    caller gets the live object from `record_run` and writes to it, which is
    simpler than accumulating the same facts twice and reconciling them.
    """

    run_id: str
    agent: str
    depth: int
    parent_id: str | None = None
    root_id: str | None = None
    steps: int = 0
    seconds: float = 0.0
    stopped_because: str = ""
    # Tool names in call order, including repeats. The order is the useful
    # part - "search, search, search, finish" and "list, search, finish"
    # cost the same and mean very different things.
    tools: list[str] = field(default_factory=list)

    def record_tool(self, name: str) -> None:
        """Note that `name` was dispatched. Called once per completed step."""
        self.steps += 1
        if name:
            # Empty for an unparseable reply, which still cost a step and is
            # already counted above. Naming it "" in the tool list would just
            # add noise to the one field whose value is its readability.
            self.tools.append(name)


@contextmanager
def record_run(agent: str) -> Iterator[RunRecord]:
    """
    Open a run record for `agent`, and log it when the block exits.

    The record's parentage is taken from the surrounding context, so a run
    started inside another run is automatically that run's child at one
    greater depth. Nothing is passed in.

    Always logs, including when the block raises: a run that died is the one
    you most want a line for, and the exception continues to propagate. The
    `stopped_because` of a failed run is left as whatever it was when the
    exception hit, which is "" for a run that never reached its own ending -
    itself the signal that it did not get there.
    """
    parent_id = _current_run_id.get()
    root_id = _root_run_id.get()

    record = RunRecord(
        run_id=uuid.uuid4().hex[:12],
        agent=agent,
        # Depth is derived from whether there is a parent at all rather than
        # from a counter, so it cannot drift out of step with the parentage
        # it is meant to describe.
        depth=0 if parent_id is None else _depth_of(parent_id) + 1,
        parent_id=parent_id,
    )
    record.root_id = root_id or record.run_id

    current_token = _current_run_id.set(record.run_id)
    root_token = _root_run_id.set(record.root_id)
    _depths[record.run_id] = record.depth

    started = time.perf_counter()
    try:
        yield record
    finally:
        record.seconds = round(time.perf_counter() - started, 3)
        _current_run_id.reset(current_token)
        _root_run_id.reset(root_token)
        _depths.pop(record.run_id, None)
        _emit(record)


# Depth by run id, for the lifetime of that run only - entries are removed in
# `record_run`'s finally. A plain dict rather than a third ContextVar because
# it is only ever read for an id that is currently on the stack, and a
# dict keyed by id says that more directly than a stack of depths would.
_depths: dict[str, int] = {}


def _depth_of(run_id: str) -> int:
    return _depths.get(run_id, 0)


def _emit(record: RunRecord) -> None:
    """
    Write the record out.

    The single place that knows the destination. Shipping these to a
    collector instead of (or as well as) the logger is a change to this
    function and to nothing else.
    """
    logger.info("agent_run %s", asdict(record), extra={"agent_run": asdict(record)})
