from copy import deepcopy

import pytest

from app.mcp_identity import IdentityError, sign_context, verify_context

SECRET = "a" * 64
ARGS = {"date": "2026-11-17"}


def test_identity_is_bound_to_owner_scope_arguments_and_action():
    meta = sign_context(SECRET, 101, "tool:get_schedule", ARGS, now=100)
    assert verify_context(SECRET, meta, "tool:get_schedule", ARGS, now=110) == (101, None)
    for secret, scope, arguments in [
        ("b" * 64, "tool:get_schedule", ARGS),
        (SECRET, "tool:add_lesson", ARGS),
        (SECRET, "tool:get_schedule", {"date": "2026-11-18"}),
    ]:
        with pytest.raises(IdentityError):
            verify_context(secret, meta, scope, arguments, now=110)
    for key, value in [("user_id", 202), ("action_id", "other"), ("issued_at", 99)]:
        changed = deepcopy(meta)
        changed["study_identity"][key] = value
        with pytest.raises(IdentityError):
            verify_context(SECRET, changed, "tool:get_schedule", ARGS, now=110)


@pytest.mark.parametrize("now", [94, 161])
def test_expired_or_future_identity_is_rejected(now):
    meta = sign_context(SECRET, 101, "tool:get_schedule", ARGS, now=100)
    with pytest.raises(IdentityError):
        verify_context(SECRET, meta, "tool:get_schedule", ARGS, now=now)


@pytest.mark.parametrize("meta", [None, {}, {"study_identity": {}}, {"study_identity": "bad"}])
def test_missing_identity_fails_without_exposing_values(meta):
    with pytest.raises(IdentityError) as error:
        verify_context(SECRET, meta, "tool:get_schedule", ARGS, now=100)
    assert SECRET not in str(error.value)
