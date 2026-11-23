from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult
from test_llm_tools import fixture as llm_fixture
from test_reminder_agent import fixture as reminder_fixture
from test_schedule_agent import NOW, call, decision, fixture

from app.audit import AuditTrace
from app.llm import LlmError


@pytest.mark.parametrize("action", ["respond", "clarify"])
async def test_text_decision_is_classified_structurally_without_changing_answer(action):
    agent, llm, _, _ = fixture({"role": "assistant", "content": "Публичный ответ."})
    llm.classify_answer = AsyncMock(return_value=action)
    trace = AuditTrace()
    assert (
        await agent.answer("Вопрос", system_prompt="private-prompt", audit=trace)
        == "Публичный ответ."
    )
    assert trace.steps[0].action == action
    assert trace.steps[0].execution == "success"
    assert "private-prompt" not in str(trace.steps)
    llm.classify_answer.assert_awaited_once_with("Публичный ответ.")


async def test_classifier_error_preserves_original_answer_and_records_unknown_kind():
    agent, llm, _, _ = fixture({"role": "assistant", "content": "Ответ."})
    llm.classify_answer = AsyncMock(side_effect=LlmError("private-error"))
    trace = AuditTrace()
    assert await agent.answer("Вопрос", system_prompt="Помогай", audit=trace) == "Ответ."
    assert trace.steps[0].action == "undetermined"
    assert trace.steps[0].reason == "classification_unavailable"
    assert "private-error" not in str(trace.steps)


async def test_read_success_has_safe_date_and_no_response_classifier_call():
    agent, llm, _, _ = fixture(decision(call()), {"role": "assistant", "content": "Занятий нет."})
    llm.classify_answer = AsyncMock()
    trace = AuditTrace()
    await agent.answer("Моё расписание", system_prompt="Помогай", audit=trace)
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert (step.action, step.tool, step.validation, step.execution) == (
        "tool",
        "get_schedule",
        "passed",
        "success",
    )
    assert step.arguments == {"date": "2026-11-17"}
    llm.classify_answer.assert_not_awaited()


@pytest.mark.parametrize(
    "output", [CallToolResult(content=[], isError=True), CallToolResult(content=[])]
)
async def test_mcp_failure_is_recorded_without_claiming_success(output):
    agent, _, _, _ = fixture(decision(call()), result=output)
    trace = AuditTrace()
    with pytest.raises(LlmError):
        await agent.answer("Вопрос", system_prompt="Помогай", audit=trace)
    assert trace.steps[0].execution == "error"


async def test_invalid_unknown_tool_never_records_raw_tool_or_arguments():
    agent, _, _, session = fixture(
        decision(call(name="private-token", arguments={"secret": "value"}))
    )
    trace = AuditTrace()
    with pytest.raises(LlmError):
        await agent.answer("Вопрос", system_prompt="Помогай", audit=trace)
    assert trace.steps[0].tool is None
    assert trace.steps[0].validation == "rejected"
    assert trace.steps[0].arguments == {}
    assert "private-token" not in str(trace.steps)
    session.call_tool.assert_not_awaited()


async def test_two_reads_have_individual_execution_status_and_limit_event():
    agent, _, _, session = fixture(
        decision(call(call_id="a"), call(call_id="b")), decision(call(call_id="c"))
    )
    trace = AuditTrace()
    with pytest.raises(LlmError, match="лимит"):
        await agent.answer("Вопрос", system_prompt="Помогай", audit=trace)
    assert session.call_tool.await_count == 2
    assert [step.execution for step in trace.steps] == ["success", "success", "limit_exceeded"]


async def test_failed_final_llm_response_preserves_successful_read_as_separate_step():
    agent, llm, _, _ = fixture(decision(call()))
    llm.complete_with_tools.side_effect = [decision(call()), LlmError("private-error")]
    trace = AuditTrace()
    with pytest.raises(LlmError):
        await agent.answer("Вопрос", system_prompt="Помогай", audit=trace)
    assert [step.execution for step in trace.steps] == ["success", "error"]
    assert trace.steps[-1].action == "undetermined"


async def test_reminder_proposal_logs_time_and_text_length_but_not_private_text():
    agent, _, mcp, _ = reminder_fixture()
    trace = AuditTrace()
    pending = await agent.answer(
        "Напомни",
        system_prompt="Помогай",
        user_id=101,
        update_id=55,
        audit=trace,
        now=NOW,
    )
    step = trace.steps[0]
    assert step.execution == "awaiting_confirmation"
    assert step.arguments["text_length"] == len(pending.payload.text)
    assert pending.payload.text not in str(trace.steps)
    assert "user_id" not in str(step.arguments)
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("action", ["respond", "clarify"])
async def test_classifier_uses_separate_json_schema_request_without_tools(action):
    llm, session = llm_fixture({"content": '{"action":"' + action + '"}'}, finish="stop")
    assert await llm.classify_answer("Публичный ответ") == action
    payload = session.post.call_args.kwargs["json"]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in payload
    assert payload["messages"][-1] == {"role": "user", "content": "Публичный ответ"}


@pytest.mark.parametrize(
    "body", ['{"action":"delete"}', '{"action":"respond","reason":"secret"}', "plain text"]
)
async def test_invalid_classifier_output_is_rejected_without_raw_text(body):
    llm, _ = llm_fixture({"content": body}, finish="stop")
    with pytest.raises(LlmError) as error:
        await llm.classify_answer("Текст")
    assert body not in str(error.value)
