from src.data.openvla_action_utils import (
    OPENVLA_ACTION_PROMPT_TEMPLATE,
    format_openvla_action_prompt,
)


def test_format_openvla_action_prompt_uses_official_template() -> None:
    prompt = format_openvla_action_prompt("Pick up the blue block.")

    assert prompt == "In: What action should the robot take to pick up the blue block?\nOut:"


def test_format_openvla_action_prompt_supports_instruction_raw_placeholder() -> None:
    prompt = format_openvla_action_prompt(
        "  Turn on the stove.  ",
        template="USER: {instruction_raw}\nASSISTANT:",
    )

    assert prompt == "USER: Turn on the stove.\nASSISTANT:"


def test_openvla_prompt_template_constant_is_explicit() -> None:
    assert OPENVLA_ACTION_PROMPT_TEMPLATE == "In: What action should the robot take to {instruction}?\nOut:"
