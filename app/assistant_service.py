"""Прикладной вход: прежние режимы без инструментов либо новые специалисты."""

from collections.abc import Sequence
from datetime import UTC, datetime

from app.agent_contracts import DialogueTurn, user_request
from app.audit import AuditTrace
from app.history import HistoryMessage
from app.llm import LlmClient
from app.personal_schedule import PendingLesson
from app.reminders import PendingReminder, ReminderStore
from app.supervisor import Supervisor
from app.tracing import current_trace, span


class AssistantService:
    def __init__(
        self, llm: LlmClient, supervisor: Supervisor, reminders: ReminderStore, timezone: str
    ):
        self._llm, self._supervisor, self._reminders = llm, supervisor, reminders
        self.timezone = timezone

    async def answer(
        self,
        text: str,
        *,
        mode_name: str,
        system_prompt: str,
        examples: tuple[tuple[str, str], ...] = (),
        history: Sequence[HistoryMessage] = (),
        temperature: float = 0.3,
        user_id: int | None = None,
        update_id: int | None = None,
        audit: AuditTrace | None = None,
    ) -> str | PendingLesson | PendingReminder:
        if mode_name in {"translate", "quiz"}:
            return await self._llm.generate(
                text,
                system_prompt=system_prompt,
                examples=examples,
                history=history,
                temperature=temperature,
            )
        if user_id is None:
            return "Не удалось определить пользователя."
        timezone = await self._reminders.get_timezone(user_id)
        request = user_request(
            actor_id=user_id, text=text, now=datetime.now(UTC), timezone=timezone or self.timezone
        )
        request = request.model_copy(
            update={
                "history": tuple(
                    DialogueTurn(role=turn["role"], content=turn["content"][:4000])
                    for turn in history[-4:]
                )
            }
        )
        trace = current_trace.get()
        if trace is not None:
            request = request.model_copy(
                update={"trace_id": trace.id, "request_id": trace.request_id}
            )
        with span("supervisor", "handle") as step:
            response = await self._supervisor.handle(request, update_id=update_id)
            if trace is not None:
                trace.route = response.route.target
                trace.confidence = response.route.confidence
            if response.result is not None:
                step.status = {
                    "success": "ok",
                    "needs_clarification": "clarify",
                    "not_found": "not_found",
                    "failed": "failed",
                }[response.result.status]
            elif response.route.target == "clarify" and not response.route.dialogue:
                step.status = "clarify"
            elif not response.route.dialogue and response.pending_lesson is None:
                step.status = "failed"
        if response.route.dialogue:
            return await self._llm.generate(
                text,
                system_prompt=system_prompt,
                examples=examples,
                history=history,
                temperature=temperature,
            )
        if audit is not None:
            if response.result and response.result.used_tools:
                for name in response.result.used_tools:
                    audit.add("tool", tool=name, validation="passed", execution="success")
            else:
                audit.add("clarify", execution="success", reason="missing_details")
        return response.pending_lesson or response.pending_reminder or response.message
