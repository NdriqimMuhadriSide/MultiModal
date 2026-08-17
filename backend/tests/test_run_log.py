"""
Tests for the per-run records agents emit (agents/run_log.py).

What matters here is parentage. A record per run is easy; a record that
knows which run caused it is the thing that makes a tree readable, and it is
tracked through a context variable rather than a parameter, so it is worth
pinning down that the variable is set and reset where it should be.
"""
import logging

import pytest

from agents.run_log import record_run


@pytest.fixture
def records(caplog):
    """
    Returns the RunRecord dicts emitted so far, in the order they were
    logged - which is completion order, so a child appears before the parent
    that was waiting on it.
    """
    caplog.set_level(logging.INFO, logger="agents.run_log")

    def collected() -> list[dict]:
        return [
            entry.agent_run for entry in caplog.records if hasattr(entry, "agent_run")
        ]

    return collected


def test_a_run_emits_one_record_with_what_it_did(records):
    with record_run("research") as run:
        run.record_tool("search")
        run.record_tool("finish")
        run.stopped_because = "finished"

    assert len(records()) == 1
    assert records()[0]["agent"] == "research"
    assert records()[0]["steps"] == 2
    assert records()[0]["tools"] == ["search", "finish"]
    assert records()[0]["stopped_because"] == "finished"
    assert records()[0]["seconds"] >= 0


def test_a_nested_run_records_its_parent_and_depth(records):
    with record_run("supervisor") as parent:
        parent.record_tool("research_documents")
        with record_run("research") as child:
            child.record_tool("search")

    # Children finish first, so they are logged first.
    child_record, parent_record = records()

    assert child_record["agent"] == "research"
    assert child_record["depth"] == 1
    assert child_record["parent_id"] == parent_record["run_id"]

    assert parent_record["depth"] == 0
    assert parent_record["parent_id"] is None


def test_a_whole_tree_shares_one_root_id(records):
    """
    What makes a tree greppable with one id, rather than by walking parent
    links back up from every leaf.
    """
    with record_run("supervisor"):
        with record_run("research"):
            with record_run("nested-deeper"):
                pass

    root_ids = {record["root_id"] for record in records()}
    assert len(root_ids) == 1
    assert [record["depth"] for record in records()] == [2, 1, 0]


def test_a_second_root_run_does_not_inherit_the_first_ones_parentage(records):
    """
    The context variable is reset on the way out. Without that, two requests
    served on the same thread would look like one tree.
    """
    with record_run("supervisor"):
        pass
    with record_run("supervisor"):
        pass

    first, second = records()
    assert first["parent_id"] is None and second["parent_id"] is None
    assert first["root_id"] != second["root_id"]
    assert second["depth"] == 0


def test_a_run_that_raises_is_still_logged(records):
    """
    A run that died is the one you most want a line for, and the exception
    still propagates.
    """
    with pytest.raises(RuntimeError):
        with record_run("supervisor") as run:
            run.record_tool("search")
            raise RuntimeError("the provider went away")

    assert len(records()) == 1
    assert records()[0]["tools"] == ["search"]
    # Never reached its own ending, and says so by not claiming one.
    assert records()[0]["stopped_because"] == ""


def test_an_unparseable_reply_counts_as_a_step_without_naming_a_tool(records):
    """
    It cost a turn, so it counts; it named no tool, so the tool list stays
    readable rather than filling with empty strings.
    """
    with record_run("research") as run:
        run.record_tool("")
        run.record_tool("search")

    assert records()[0]["steps"] == 2
    assert records()[0]["tools"] == ["search"]
