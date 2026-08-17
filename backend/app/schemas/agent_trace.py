"""
The trace shape every hand-written agent returns.

Shared by the research agent and the vision agent, because the trace is a
property of the loop (agents/agent_loop.py), not of what any particular
agent does with it. Two identical models would let the two endpoints drift
apart field by field, and the frontend would then need two renderers for
the same thing.
"""
import json

from pydantic import BaseModel, ConfigDict, Field


class AgentStepModel(BaseModel):
    """
    One turn of an agent's loop, as shown to the client.

    The trace is part of the response rather than a debug-only extra. These
    agents reach their answers through several tool calls the user never
    asked for and cannot see, and "trust me, I looked" is not a reasonable
    thing to ask - the steps are how a reader checks that the right things
    were done, and how they tell a thorough answer apart from a lucky one.
    """

    model_config = ConfigDict(populate_by_name=True)

    thought: str
    tool: str
    # The arguments the agent chose, as JSON text rather than a typed
    # object: every tool takes a different shape, and a union of per-tool
    # models would have to change every time a tool is added, for something
    # no client branches on.
    tool_input: str = Field(..., alias="toolInput")
    observation: str
    # The steps a delegated specialist took to produce this step's
    # observation. Empty for every ordinary tool, so a client that predates
    # delegation - and every trace from an agent that does not delegate -
    # sees exactly what it saw before.
    #
    # Recursive rather than one flat level, because nothing in the loop caps
    # how deep delegation goes and a schema that assumed a depth would be
    # wrong the first time it changed. The renderer recurses; the model
    # simply says so.
    children: list["AgentStepModel"] = Field(default_factory=list)


def to_step_models(steps) -> list[AgentStepModel]:
    """
    Project a run's `AgentStep` list onto the response model, children and all.

    Written once here rather than inline in each endpoint. There are now
    three callers (agent, research, vision-agent), and the projection stopped
    being a two-line comprehension the moment it had to recurse - three
    hand-rolled recursions would be three chances for one endpoint to stop
    reporting nested steps and nobody to notice.
    """
    return [
        AgentStepModel(
            thought=step.thought,
            tool=step.tool,
            tool_input=json.dumps(step.tool_input),
            observation=step.observation,
            children=to_step_models(step.children),
        )
        for step in steps
    ]


def to_step_payload(step) -> dict:
    """
    Project one step onto the shape a streamed SSE frame carries.

    Byte-identical to an entry in the corresponding non-streaming response -
    `toolInput` rather than `tool_input`, arguments as JSON text, children
    nested the same way - so a client has one step type whether it streamed
    the run or not.
    """
    return {
        "thought": step.thought,
        "tool": step.tool,
        "toolInput": json.dumps(step.tool_input),
        "observation": step.observation,
        "children": [to_step_payload(child) for child in step.children],
    }
