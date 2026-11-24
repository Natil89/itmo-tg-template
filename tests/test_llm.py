import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from app.history import DEFAULT_HISTORY_LIMITS, HistoryLimits
from app.llm import LlmClient, LlmError


def make_client(*, status=200, payload=None, failure=None, history_limits=DEFAULT_HISTORY_LIMITS):
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=payload)
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response, side_effect=failure)
    request.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = request
    client = LlmClient(
        session, api_key="test-key", model="test/model", history_limits=history_limits
    )
    return client, session, response


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_request_preserves_roles_prompt_and_parameters(temperature):
    # Arrange
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]}
    client, session, _ = make_client(payload=payload)
    # Act
    answer = await client.generate("Вопрос", system_prompt="Инструкция", temperature=temperature)
    # Assert
    assert answer == "Ответ"
    assert session.post.call_args.args == ("https://openrouter.ai/api/v1/chat/completions",)
    request = session.post.call_args.kwargs
    assert request["json"] == {
        "model": "test/model",
        "messages": [
            {"role": "system", "content": "Инструкция"},
            {"role": "user", "content": "Вопрос"},
        ],
        "temperature": temperature,
        "reasoning": {"enabled": False},
        "max_tokens": 1000,
    }
    assert request["allow_redirects"] is False


@pytest.mark.parametrize("status", [302, 401, 402, 403, 429, 500])
async def test_http_errors_do_not_expose_response_body(status):
    # Arrange
    client, _, response = make_client(status=status, payload={"error": "test-key"})
    # Act / Assert
    with pytest.raises(LlmError) as error:
        await client.generate("Вопрос")
    assert "test-key" not in str(error.value)
    response.json.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"error": {"message": "test-key"}},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": " "}}]},
        {"choices": [{"finish_reason": "length", "message": {"content": "часть"}}]},
        {"choices": [{"finish_reason": "content_filter", "message": {"content": "часть"}}]},
    ],
)
async def test_invalid_or_incomplete_answer_is_not_returned(payload):
    # Arrange
    client, _, _ = make_client(payload=payload)
    # Act / Assert
    with pytest.raises(LlmError) as error:
        await client.generate("Вопрос")
    assert "test-key" not in str(error.value)


@pytest.mark.parametrize(
    "failure", [TimeoutError("test-key"), aiohttp.ClientConnectionError("test-key")]
)
async def test_network_errors_have_safe_messages(failure):
    # Arrange
    client, _, _ = make_client(failure=failure)
    # Act / Assert
    with pytest.raises(LlmError) as error:
        await client.generate("Вопрос")
    assert "test-key" not in str(error.value)


async def test_invalid_json_has_safe_error():
    # Arrange
    client, _, response = make_client()
    response.json.side_effect = ValueError("test-key")
    # Act / Assert
    with pytest.raises(LlmError, match="некорректный ответ"):
        await client.generate("Вопрос")


@pytest.mark.parametrize("text,temperature", [(" ", 0.3), ("Вопрос", 0.5), ("Вопрос", True)])
async def test_invalid_input_is_rejected_before_request(text, temperature):
    # Arrange
    client, session, _ = make_client()
    # Act / Assert
    with pytest.raises(LlmError):
        await client.generate(text, temperature=temperature)
    session.post.assert_not_called()


async def test_cancellation_propagates():
    # Arrange
    client, _, _ = make_client(failure=asyncio.CancelledError())
    # Act / Assert
    with pytest.raises(asyncio.CancelledError):
        await client.generate("Вопрос")


async def test_few_shot_examples_keep_roles_and_precede_actual_question():
    # Arrange
    from app.modes import MODES

    mode = MODES["translate"]
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "Hello!"}}]}
    client, session, _ = make_client(payload=payload)
    # Act
    await client.generate(
        "Переведи на английский: Привет!", system_prompt=mode.system_prompt, examples=mode.examples
    )
    # Assert
    messages = session.post.call_args.kwargs["json"]["messages"]
    assert messages[0] == {"role": "system", "content": mode.system_prompt}
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    for index, (question, answer) in enumerate(mode.examples):
        assert messages[1 + index * 2] == {"role": "user", "content": question}
        assert messages[2 + index * 2] == {"role": "assistant", "content": answer}
    assert messages[-1] == {"role": "user", "content": "Переведи на английский: Привет!"}


@pytest.mark.parametrize(
    "limits",
    [HistoryLimits(max_messages=3), HistoryLimits(context_budget=1102)],
)
async def test_history_keeps_latest_whole_turn_after_examples(limits):
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]}
    client, session, _ = make_client(payload=payload, history_limits=limits)
    history = [
        {"role": "user", "content": "o"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "n"},
        {"role": "assistant", "content": "b"},
    ]
    # 4 обязательных сообщения по 17 единиц, 1000 на ответ и 34 на одну пару.
    await client.generate("?", system_prompt="s", examples=(("q", "a"),), history=history)
    assert session.post.call_args.kwargs["json"]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        *history[-2:],
        {"role": "user", "content": "?"},
    ]


async def test_question_over_budget_is_rejected_before_network_request():
    client, session, _ = make_client(history_limits=HistoryLimits(context_budget=1050))
    with pytest.raises(LlmError, match="Вопрос слишком длинный"):
        await client.generate("я" * 20, system_prompt="s")
    session.post.assert_not_called()
