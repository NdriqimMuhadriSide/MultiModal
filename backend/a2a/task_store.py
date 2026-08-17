"""
Where a Task lives between the call that created it and the call that reads
it back.

WHY THIS EXISTS AT ALL IN A BLOCKING IMPLEMENTATION

`message/send` runs the agent and returns the finished Task, so a caller
that only ever sends and reads the reply never needs storage. `tasks/get`
is what needs it - and `tasks/get` is not optional in A2A, because a client
whose connection dropped mid-call has a task id and no result, and its only
recovery is to ask again by id.

That makes this the smallest honest implementation: keep finished tasks
around long enough to be re-read, and no longer.

WHY IN-MEMORY IS NOT THE PRODUCTION ANSWER

Three things break the moment there is more than one worker process, and all
three are silent rather than loud:

    a task created on worker A is not found on worker B, so `tasks/get`
    returns TaskNotFound for a task that genuinely completed

    a restart loses every task, so a caller retrying after a deploy is told
    its work never happened

    push notifications and `tasks/resubscribe` cannot be built on it at all,
    since both require the task to outlive the request that made it

The interface below is the seam for fixing that: a Redis implementation is
the same three methods, and nothing above this module changes.

BOUNDED ON PURPOSE

A dict that only grows is a leak in a long-lived process, and a research
task carries an answer, a trace and a source list - not bytes, but not
nothing either. Eviction is oldest-first past a ceiling, which is the right
policy here because a task's value drops to almost zero once its caller has
read it, and the caller reads it immediately or not at all.
"""
import threading
from collections import OrderedDict

from a2a.types import Task


class TaskStore:
    """
    The interface server.py depends on.

    Three methods, deliberately - `save`, `get`, and nothing that iterates.
    A store that can list its tasks invites a client to enumerate other
    callers' work, and no A2A method needs it.
    """

    def save(self, task: Task) -> None:
        raise NotImplementedError

    def get(self, task_id: str) -> Task | None:
        raise NotImplementedError


class InMemoryTaskStore(TaskStore):
    """
    A bounded, thread-safe dict of tasks.

    Thread-safe rather than not, because FastAPI runs synchronous routes in a
    threadpool: two `message/send` calls genuinely execute concurrently here,
    and an OrderedDict mutated from two threads can lose an entry or raise
    during iteration. The lock is held only around the dict operations, never
    around an agent run.
    """

    def __init__(self, max_tasks: int = 256) -> None:
        self._tasks: OrderedDict[str, Task] = OrderedDict()
        self._max_tasks = max_tasks
        self._lock = threading.Lock()

    def save(self, task: Task) -> None:
        with self._lock:
            # Re-inserting an existing id moves it to the end, so a task that
            # is still being updated is not evicted while in use.
            self._tasks.pop(task.id, None)
            self._tasks[task.id] = task
            while len(self._tasks) > self._max_tasks:
                self._tasks.popitem(last=False)

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)


# One store per process, shared by every request.
#
# A module-level singleton rather than a FastAPI dependency with state,
# because the whole point is that it outlives the request that wrote to it -
# a per-request store would make `tasks/get` unable to find anything, which
# is a bug that only shows up under the exact conditions the method exists
# for.
_store: InMemoryTaskStore | None = None


def get_task_store() -> TaskStore:
    """Return the process-wide task store, creating it on first use."""
    global _store
    if _store is None:
        from app.core.config import settings

        _store = InMemoryTaskStore(max_tasks=settings.a2a_max_stored_tasks)
    return _store
