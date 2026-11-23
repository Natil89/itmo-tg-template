import pytest

from app.formatting import format_answer, utf16_length


def selected_text(message, entity):
    encoded = message.text.encode("utf-16-le")
    return encoded[entity.offset * 2 : (entity.offset + entity.length) * 2].decode("utf-16-le")


def test_headings_bold_italic_and_inline_code():
    # Arrange / Act
    messages = format_answer("# Заголовок\n\n**Жирный**, *курсив* и `x = 1`.")
    # Assert
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "Заголовок\n\nЖирный, курсив и x = 1."
    assert {(entity.type, selected_text(message, entity)) for entity in message.entities} == {
        ("bold", "Заголовок"),
        ("bold", "Жирный"),
        ("italic", "курсив"),
        ("code", "x = 1"),
    }


def test_python_block_preserves_indentation_operators_and_markdown_characters():
    # Arrange
    code = 'if x < 3:\n    print("**текст** & <b>")\n'
    # Act
    message = format_answer("```python\n" + code + "```\n")[0]
    # Assert
    assert message.text == code
    assert len(message.entities) == 1
    assert message.entities[0].type == "pre"
    assert message.entities[0].language == "python"
    assert selected_text(message, message.entities[0]) == code


def test_lists_and_links():
    # Arrange / Act
    message = format_answer("- один\n- два\n\n3. три\n4. четыре\n\n[Сайт](https://example.org)")[0]
    # Assert
    assert message.text == "• один\n• два\n\n3. три\n4. четыре\n\nСайт"
    link = message.entities[0]
    assert link.type == "text_link"
    assert link.url == "https://example.org"
    assert selected_text(message, link) == "Сайт"


def test_html_and_variable_underscores_remain_literal():
    # Arrange
    text = "<b>Это не HTML</b> & my_list = [1, 2]"
    # Act
    message = format_answer(text)[0]
    # Assert
    assert message.text == text
    assert not message.entities


def test_emoji_offsets_are_utf16():
    # Arrange / Act
    message = format_answer("🙂 **важно** и `🙂`")[0]
    # Assert
    bold = next(entity for entity in message.entities if entity.type == "bold")
    assert bold.offset == 3
    assert selected_text(message, bold) == "важно"
    code = next(entity for entity in message.entities if entity.type == "code")
    assert code.length == 2
    assert selected_text(message, code) == "🙂"


@pytest.mark.parametrize(
    "prefix,suffix,kind", [("**", "**", "bold"), ("```python\n", "\n```", "pre")]
)
def test_formatting_survives_chunk_boundaries(prefix, suffix, kind):
    # Arrange
    content = "🙂x" * 4000
    # Act
    messages = format_answer(prefix + content + suffix)
    # Assert
    assert len(messages) > 1
    assert "".join(message.text for message in messages) == content + (
        "\n" if kind == "pre" else ""
    )
    for message in messages:
        assert 0 < utf16_length(message.text) <= 4096
        assert len(message.entities) == 1
        assert message.entities[0].type == kind
        assert message.entities[0].offset == 0
        assert message.entities[0].length == utf16_length(message.text)


def test_inline_code_does_not_overlap_other_formatting():
    # Arrange / Act
    message = format_answer("**до `x` после**")[0]
    # Assert
    code = next(entity for entity in message.entities if entity.type == "code")
    assert selected_text(message, code) == "x"
    bold = [entity for entity in message.entities if entity.type == "bold"]
    assert [selected_text(message, entity) for entity in bold] == ["до ", " после"]


@pytest.mark.parametrize("text", ["Незакрытая **разметка", "```python\nx = 1", "***", ""])
def test_incomplete_markdown_never_produces_empty_messages(text):
    # Arrange / Act
    messages = format_answer(text)
    # Assert
    assert all(message.text for message in messages)
    if text:
        assert messages


def test_nested_emphasis_is_supported():
    # Arrange / Act
    message = format_answer("***важно***")[0]
    # Assert
    assert message.text == "важно"
    assert {entity.type for entity in message.entities} == {"bold", "italic"}
