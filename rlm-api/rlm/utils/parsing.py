"""
Parsing utilities for RLM trjaectories.
"""

import json
import re
from typing import TYPE_CHECKING, Optional

from rlm.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from rlm.environments.base_env import BaseEnv


def _extract_structured_response(text: str) -> Optional[dict]:
    """Best-effort parse for the structured JSON response format."""
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*\n", "", candidate)
        candidate = re.sub(r"\n```\s*$", "", candidate)
        candidate = candidate.strip()

    try:
        payload = json.loads(candidate)
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def _normalize_code_block(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:repl|python)?\s*\n", "", code)
        code = re.sub(r"\n```\s*$", "", code)
    return code.strip()


def _resolve_final_var(environment: "BaseEnv", variable_name: str) -> Optional[str]:
    """Resolve a variable via FINAL_VAR and return its value when available."""
    if not variable_name:
        return None

    result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
    final_answer = result.stdout.strip()
    if not final_answer:
        final_answer = result.stderr.strip() or ""

    if not final_answer or final_answer.startswith("Error: Variable"):
        return None
    return final_answer


def _resolve_fallback_summary_var(environment: "BaseEnv") -> Optional[str]:
    """Fallback for malformed structured outputs that omit variable_with_result."""
    probe_code = (
        "for _name in ['final_summary', 'summary', 'final_answer', 'answer', 'result']:\n"
        "    if _name in locals():\n"
        "        _value = FINAL_VAR(_name)\n"
        "        if isinstance(_value, str) and not _value.startswith('Error: Variable'):\n"
        "            print(_name)\n"
        "            break"
    )
    probe_result = environment.execute_code(probe_code)
    candidate = probe_result.stdout.strip().splitlines()
    if not candidate:
        return None

    return _resolve_final_var(environment, candidate[-1].strip())


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    structured_response = _extract_structured_response(text)
    if structured_response is not None:
        code_blocks = structured_response.get("code_blocks", [])
        if isinstance(code_blocks, list):
            return [
                _normalize_code_block(str(code_block))
                for code_block in code_blocks
                if str(code_block).strip()
            ]

    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def find_final_answer(text: str, environment: Optional["BaseEnv"] = None) -> Optional[str]:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return the final answer string.

    If FINAL_VAR is found and an environment is provided, executes code to retrieve the variable value.
    Returns None if neither pattern is found.

    Args:
        text: The response text to parse
        environment: Optional environment to execute code for FINAL_VAR retrieval

    Returns:
        The final answer string, or None if no final answer pattern is found
    """
    structured_response = _extract_structured_response(text)
    if structured_response is not None and structured_response.get("done") is True:
        variable_name = str(structured_response.get("variable_with_result", "")).strip()
        if environment is not None:
            if variable_name:
                final_answer = _resolve_final_var(environment, variable_name)
                if final_answer is not None:
                    return final_answer
            return None

    # Check for FINAL_VAR pattern first - must be at start of line
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is not None:
            return _resolve_final_var(environment, variable_name)
        return None

    # Check for FINAL pattern - must be at start of line
    # Use greedy matching to capture content with nested parentheses
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger) -> Optional[str]:
    """Check if response contains a final answer."""
    # Use the new find_final_answer function which handles both FINAL and FINAL_VAR
    return find_final_answer(response, environment=repl_env)


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
