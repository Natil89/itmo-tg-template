import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm import LlmClient, LlmError


def fixture(message, finish="tool_calls", status=200):
    response = MagicMock(status=status)
    response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": message,
                }
            ]
        }
    )
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = request
    return LlmClient(session, api_key="fake-key", model="test/model"), session


async def test_native_tool_call_uses_provider_contract():
    call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "get_schedule",
            "arguments": json.dumps({"date": "2026-11-17"}),
        },
    }
    llm, session = fixture({"tool_calls": [call], "content": None})
    tools = [{"type": "function", "function": {"name": "get_schedule"}}]
    answer = await llm.complete_with_tools([{"role": "user", "content": "Завтра?"}], tools)
    assert answer == {"role": "assistant", "content": None, "tool_calls": [call]}
    payload = session.post.call_args.kwargs["json"]
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False


async def test_provider_index_is_accepted_but_not_forwarded_to_model():
    # Arrange: форма реального ответа настроенного провайдера OpenRouter.
    call = {
        "id": "call-1",
        "type": "function",
        "index": 0,
        "function": {"name": "get_schedule", "arguments": '{"date":"2026-11-16"}'},
    }
    llm, _ = fixture({"tool_calls": [call], "content": None})
    # Act
    answer = await llm.complete_with_tools([], [])
    # Assert: служебный индекс не меняет вызов и не передаётся в следующий turn.
    assert answer["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": call["function"],
        }
    ]


@pytest.mark.parametrize("index", ["0", -1, True, 0.5])
async def test_malformed_provider_index_still_rejected(index):
    call = {
        "id": "call-1",
        "type": "function",
        "index": index,
        "function": {"name": "get_schedule", "arguments": "{}"},
    }
    llm, _ = fixture({"tool_calls": [call], "content": None})
    with pytest.raises(LlmError, match="некорректный вызов"):
        await llm.complete_with_tools([], [])


async def test_tool_call_diagnostics_do_not_log_values_or_unknown_field_names(caplog):
    call = {
        "id": "call-1",
        "type": "function",
        "private-key": "private-value",
        "function": {"name": "get_schedule", "arguments": "private-arguments"},
    }
    llm, _ = fixture({"tool_calls": [call], "content": None})
    with pytest.raises(LlmError):
        await llm.complete_with_tools([], [])
    assert "extra_forbidden" in caplog.text
    assert "private-key" not in caplog.text
    assert "private-value" not in caplog.text
    assert "private-arguments" not in caplog.text


@pytest.mark.parametrize(
    "call",
    [
        {"id": "x", "type": "function", "function": {"name": "tool", "arguments": {}}},
        {
            "id": "x",
            "type": "function",
            "function": {"name": "tool", "arguments": "{}"},
            "extra": "fake-key",
        },
        {"type": "function", "function": {"name": "tool", "arguments": "{}"}},
    ],
)
async def test_malformed_native_calls_are_rejected_without_raw_content(call):
    llm, _ = fixture({"tool_calls": [call]})
    with pytest.raises(LlmError) as error:
        await llm.complete_with_tools([], [])
    assert "fake-key" not in str(error.value)


async def test_tools_method_returns_plain_answer_without_tools():
    llm, session = fixture({"content": "Ответ"}, finish="stop")
    assert await llm.complete_with_tools([], []) == {"role": "assistant", "content": "Ответ"}
    assert "tools" not in session.post.call_args.kwargs["json"]


async def test_tools_http_error_is_safe_and_uses_shared_transport():
    llm, _ = fixture({"content": "fake-key"}, status=401)
    with pytest.raises(LlmError, match="Нет доступа"):
        await llm.complete_with_tools([], [])


async def test_history_near_budget_is_shortened_without_dropping_current_question():
    from app.history import HistoryLimits

    # Arrange: legacy estimate fits, but serialized roles/JSON add more overhead.
    llm, session = fixture({"content": "Ответ"}, finish="stop")
    llm._history_limits = HistoryLimits(context_budget=4000)
    messages = llm.build_messages(
        "x",
        system_prompt="s",
        tools=[],
        history=[
            {"role": "user", "content": "a" * 1460},
            {"role": "assistant", "content": "b" * 1460},
        ],
    )
    # Act
    result = await llm.complete_with_tools(messages, [])
    # Assert: transport still called; only the old complete pair was removed.
    assert result["content"] == "Ответ"
    assert session.post.call_args.kwargs["json"]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x"},
    ]


async def test_tool_result_growth_removes_old_history_but_preserves_current_exchange():
    from app.history import HistoryLimits

    llm, session = fixture({"content": "Ответ"}, finish="stop")
    llm._history_limits = HistoryLimits(context_budget=4000)
    messages = llm.build_messages(
        "current",
        system_prompt="s",
        tools=[],
        examples=(("example", "example-answer"),),
        history=[
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 1000},
        ],
    )
    assert len(messages) == 6
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c",
                "type": "function",
                "function": {
                    "name": "get_schedule",
                    "arguments": "{}",
                },
            }
        ],
    }
    tool_result = {"role": "tool", "tool_call_id": "c", "content": "r" * 1000}
    messages.extend([tool_call, tool_result])
    await llm.complete_with_tools(messages, [])
    sent = session.post.call_args.kwargs["json"]["messages"]
    assert sent == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "example-answer"},
        {"role": "user", "content": "current"},
        tool_call,
        tool_result,
    ]
