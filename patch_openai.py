"""
patch_openai.py
===============
Monkey-patches a `responses.create()` method onto any OpenAI client instance
that lacks a native /v1/responses endpoint (e.g. llama.cpp).

All returned objects are genuine openai.types.responses SDK types so callers
get correct attribute access, type-checking, and isinstance() behaviour. Hopefully.

It's rough and ready but should is useful here and there where there's no support for 'responses'

Example Usage
-----
    from patch_openai import monkey_patch_responses_api
    llm = OpenAI(base_url=LLAMA_SERVER, api_key=os.environ.get("LLAMA_API_KEY"))
    monkey_patch_responses_api(llm)          # llm is an openai.OpenAI instance


    Now llm_client has a 'responses.create' function: for example:

        response = llm_client.responses.create(model=LLAMA_MODEL, temperature=2, input="In one sentence, tell me about Stan Laurel")
        print(response.output_text)

    Remember: It's a fake endpoint so you are not really talking to responses.create() but to chat.create()
    The original code was created by claudeai but I've modified to work better. Well, to work at all but it saved a
    lot of typing initially! This current version actually handles tools.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Import the real SDK return types we need to construct
# ---------------------------------------------------------------------------
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseTextConfig,
)
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    ResponseUsage,
    InputTokensDetails,
    OutputTokensDetails,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str = "resp") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _role_to_chat_completion_role(role: str) -> str:
    """
    Map Responses-API roles to chat-completions roles - llama.cpp gets snippy
    about 'developer' (at least my one does).
    """
    return "system" if role == "developer" else role


def _input_to_messages(input_val: Any) -> list[dict]:
    """
    Convert the `input` parameter of responses.create() into a list of
    chat-completions messages.

    Handles:
      - a plain string prompt
      - a list that may contain:
          • {"role": ..., "content": ...}   — normal message dicts
          • {"type": "function_call_output", "call_id": ..., "output": ...}
          • ResponseFunctionToolCall objects  (already-appended output items)
          • ResponseOutputMessage objects
    """
    if isinstance(input_val, str):
        return [{"role": "user", "content": input_val}]

    messages: list[dict] = []
    for item in input_val:
        # ---- dict items ------------------------------------------------
        if isinstance(item, dict):
            item_type = item.get("type")
            item_role = item.get("role", "")

            if item_type == "function_call_output":
                # Responses API tool result  →  chat completions "tool" message
                messages.append({
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                })

            elif item_type == "function_call":
                # A function-call dict fed back into history — reconstruct as
                # an assistant message that contains a tool_call
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id", _make_id("call")),
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        },
                    }],
                })

            elif item_type == "message" or item_role:
                # Plain message or typed message dict
                messages.append({
                    "role": _role_to_chat_completion_role(item_role or item.get("role", "user")),
                    "content": item.get("content", ""),
                })

            else:
                # Fallback — just pass through if it already looks like a
                # chat-completions message
                messages.append(item)

        # ---- SDK output objects ----------------------------------------
        elif isinstance(item, ResponseFunctionToolCall):
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }],
            })

        elif isinstance(item, ResponseOutputMessage):
            # Flatten content blocks to a single string
            text_parts = [c.text for c in (item.content or []) if hasattr(c, "text")]
            messages.append({
                "role": "assistant",
                "content": " ".join(text_parts),
            })

        else:
            # Unknown type — attempt dict conversion as a last resort
            try:
                d = dict(item)
                messages.append(d)
            except Exception:  # Naughty but simple # noqa
                pass

    return messages


def tools_to_convert(tools: list[dict]) -> list[dict]:
    """
    Convert Responses-API tool dicts to chat-completions tool dicts.

    Responses API:
        {"type": "function", "name": "foo", "description": "...", "parameters": {...}}

    Chat Completions API:
        {"type": "function", "function": {"name": "foo", "description": "...", "parameters": {...}}}
    """
    converted_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            if "function" in tool:
                # Already in converted format — pass through
                converted_tools.append(tool)
            else:
                converted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })
        else:
            # Non-function tools (web_search, etc.) — pass through unchanged;
            # llama.cpp will ignore unknown tool types gracefully... for some
            # values of gracefully; YMMV
            converted_tools.append(tool)
    return converted_tools


def _build_response(completion: Any, model: str) -> Response:
    """
    Turn a chat-completions ChatCompletion object into a genuine
    openai.types.responses.Response object.
    """
    resp_id = _make_id("resp")
    created_at = float(getattr(completion, "created", time.time()))
    choice = completion.choices[0]
    msg = choice.message
    output_items: list[Any] = []

    # ---- tool calls (what larks!) -----------------------------------------
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            fn = tc.function
            output_items.append(
                ResponseFunctionToolCall(
                    id=_make_id("fc"),
                    call_id=tc.id,
                    name=fn.name,
                    arguments=fn.arguments,
                    type="function_call",
                    status="completed",
                )
            )

    # ---- normal assistant message ----------------------------------------
    content_text = msg.content or ""
    if content_text or not output_items:
        text_block = ResponseOutputText(
            type="output_text",
            text=content_text,
            annotations=[],
        )
        output_items.append(
            ResponseOutputMessage(
                id=_make_id("msg"),
                type="message",
                role="assistant",
                content=[text_block],
                status="completed",
            )
        )

    # ---- usage ------------------------------------------------------------
    raw_usage = getattr(completion, "usage", None)
    blank_in = InputTokensDetails(cached_tokens=0)
    blank_out = OutputTokensDetails(reasoning_tokens=0)
    if raw_usage:
        usage = ResponseUsage(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0),
            output_tokens=getattr(raw_usage, "completion_tokens", 0),
            total_tokens=getattr(raw_usage, "total_tokens", 0),
            input_tokens_details=blank_in,
            output_tokens_details=blank_out,
        )
    else:
        usage = ResponseUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_tokens_details=blank_in,
            output_tokens_details=blank_out,
        )

    # Use model_construct() to bypass Pydantic validation for fields we're
    # fabricating — this is how the OpenAI SDK itself builds response objects
    # from raw API data, and avoids breakage when field types are strict
    # Pydantic sub-models (e.g. reasoning, truncation, tool_choice).
    return Response.model_construct(
        id=resp_id,
        object="response",
        created_at=created_at,
        model=model,
        output=output_items,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        temperature=None,
        top_p=None,
        truncation="disabled",
        status="completed",
        usage=usage,
        error=None,
        incomplete_details=None,
        instructions=None,
        max_output_tokens=None,
        metadata={},
        previous_response_id=None,
        reasoning=None,
        store=False,
        text=ResponseTextConfig(format={"type": "text"}),
    )


# ---------------------------------------------------------------------------
# The slim "responses" namespace we attach to the client
# ---------------------------------------------------------------------------

class _ResponsesNamespace:
    """
    Minimal stub that exposes responses.create() backed by chat.completions.
    Yes, yes, 'input' is a reserved word, but it's needed for compatibility.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, *, model: str, input: Any, tools: list[dict] | None = None,  # noqa
               temperature: float | None = None, max_output_tokens: int | None = None, **kwargs: Any) -> Response:  # noqa
        messages = _input_to_messages(input)

        kwargs_to_build_request: dict[str, Any] = dict(model=model, messages=messages)
        if tools:
            kwargs_to_build_request["tools"] = tools_to_convert(tools)
            kwargs_to_build_request["tool_choice"] = "auto"
        if temperature is not None:
            kwargs_to_build_request["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs_to_build_request["max_tokens"] = max_output_tokens

        completion = self._client.chat.completions.create(**kwargs_to_build_request)
        return _build_response(completion, model)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def monkey_patch_responses_api(client: Any) -> None:
    """
    Attach a `responses` attribute to *client* so that
    ``client.responses.create(...)`` works like the real Responses API,
    translated transparently to ``client.chat.completions.create(...)``.

    Parameters
    ----------
    client : openai.OpenAI (or compatible)
        The client instance to patch.  Modified in-place.
    """
    client.responses = _ResponsesNamespace(client)
