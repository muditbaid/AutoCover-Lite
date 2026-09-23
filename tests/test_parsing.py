import pytest

from autocover.llm.parsing import extract_code, extract_json


def test_extract_code_prefers_longest_python_block():
    reply = "Here:\n```python\nx = 1\n```\nand\n```python\ndef f():\n    return 2\n```\n"
    assert extract_code(reply) == "def f():\n    return 2\n"


def test_extract_code_falls_back_to_untagged_then_raw():
    assert extract_code("```\nprint(1)\n```") == "print(1)\n"
    assert extract_code("print(2)") == "print(2)\n"


def test_extract_code_ignores_other_languages():
    reply = "```bash\npip install x\n```\n```py\nimport x\n```"
    assert extract_code(reply) == "import x\n"


@pytest.mark.parametrize(
    "reply",
    [
        '{"a": 1}',
        'Sure!\n```json\n{"a": 1}\n```',
        'The answer is {"a": 1} as requested.',
    ],
)
def test_extract_json_variants(reply):
    assert extract_json(reply) == {"a": 1}


def test_extract_json_list():
    assert extract_json("scenarios: [1, 2]") == [1, 2]


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json here")
