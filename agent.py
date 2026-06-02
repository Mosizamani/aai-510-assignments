import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define your LLM endpoint and system prompt (placeholder replaced by next cell)
############################################
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

SYSTEM_PROMPT = """You are the UltraFeedback Expert, an AI assistant that helps users explore and understand the UltraFeedback LLM preference dataset.

Use only the tools that are actually available to you (check the tool list). Typical tools include:
- main__default__all_model_combinations: list model pairs and comparison data from the dataset
- main__default__analyze_instruction: assess instruction complexity (word count, sentence count, etc.)
- main__default__compare_models: compare two models (e.g. win counts, average chosen/rejected ratings)
- main__default__search_similar_instructions: find similar instructions in the dataset
- main__default__find_best_technical_doc_models: find best models for technical documentation tasks
- main__default__get_best_model_source_pairs: find top-performing model-source combinations
- main__default__recommend_reasoning_agent_route: recommend model-source pairs for reasoning agents
- You.com web search: current information about LLMs, benchmarks, or research

Always cite which tool you used and explain the results. If you do not have a relevant tool, say so rather than guessing or inventing tool names."""


###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


TOOL_INFOS = []

# You can use UDFs in Unity Catalog as agent tools
# Only include functions that actually exist in main.default schema
UC_TOOL_NAMES = [
    "main.default.all_model_combinations",
    "main.default.analyze_instruction",
    "main.default.compare_models",
    "main.default.find_best_technical_doc_models",
    "main.default.get_best_model_source_pairs",
    "main.default.recommend_reasoning_agent_route",
    "main.default.search_similar_instructions",
]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Use Databricks vector search indexes as tools
# See [docs](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) for details

# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
VECTOR_SEARCH_TOOLS = []

for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))

# MCP tools are added in the notebook (see "Add MCP tools and create agent" cell)
# so that the session is not terminated when this file is imported.

def _sanitize_tool_spec(spec: dict) -> dict:
    """Remove JSON schema keywords that some model endpoints reject (e.g. minLength, minimum, minItems)."""
    import copy
    spec = copy.deepcopy(spec)
    params = spec.get("function", {}).get("parameters") or {}
    if not isinstance(params, dict) or "properties" not in params:
        return spec
    for prop in params.get("properties", {}).values():
        if not isinstance(prop, dict):
            continue
        t = prop.get("type")
        if t == "string":
            for key in ("minLength", "maxLength", "pattern"):
                prop.pop(key, None)
        elif t in ("integer", "number"):
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                prop.pop(key, None)
        elif t == "array":
            for key in ("minItems", "maxItems", "uniqueItems"):
                prop.pop(key, None)
    return spec

class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects. Strips schema keywords (e.g. minLength) that some endpoints reject."""
        return [_sanitize_tool_spec(tool_info.spec) for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        # UC rejects params like {'': '{}'} for no-arg functions; keep only valid keyed args
        sane_args = {k: v for k, v in (args or {}).items() if k and isinstance(k, str)}
        # Normalize: strip quotes and model tokens (e.g. <|channel|>commentary) appended to name
        name = tool_name.strip().strip('"').strip("'")
        if "<" in name:
            name = name.split("<")[0].strip()
        if name in self._tools_dict:
            return self._tools_dict[name].exec_fn(**sane_args)
        # Fallback: name may have extra tokens; use longest key that name starts with
        candidates = [k for k in self._tools_dict if name.startswith(k)]
        if candidates:
            return self._tools_dict[max(candidates, key=len)].exec_fn(**sane_args)
        raise KeyError(f"Unknown tool: {tool_name!r}. Known tools: {list(self._tools_dict.keys())}")

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        try:
            args = json.loads(tool_call.get("arguments"))
        except Exception as e:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 20,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)


# Log the model using MLflow (AGENT is created in the notebook after adding MCP tools)
mlflow.openai.autolog()
