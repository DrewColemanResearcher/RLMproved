from __future__ import annotations

import re
from typing import Literal

from mcp.server.fastmcp import Context

from rlm_mcp.rlm_mcp_server.stateful_mcp.session_manager import SessionManager


def register(mcp):
    @mcp.tool()
    async def find_content_using_regex(regex: str, window: int, ctx: Context):
        """
        This tool allows to find specific content inside the context variable using a regex pattern.
        It returns all the matches that were found in the context variable.

        :param regex: The regex pattern to use for finding content in the context variable.
        :param window: The window we want to read around a match
        :return: A string containing all the matches that were found in the context variable.
                 Each match is returned with a window of surrounding text for better understanding
                 of the context where the match was found. The match is in square brackets.
        """
        user_id = ctx.request_context.request.headers.get("x-user-id")
        chat_id = ctx.request_context.request.headers.get("x-chat-id")

        repl_manager = SessionManager.get_instance().get(user_id, chat_id)

        find_matches = repl_manager.execute(
            # language=Python
            f"""
def __mcp_find_regex_matches__():
    import re

    pattern = {regex!r}
    window = {int(window)}

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        print(f"Invalid regex: {{e}}")
        return

    matches = list(compiled.finditer(context))

    if not matches:
        print("No matches found.")
        return

    for i, match in enumerate(matches, 1):
        start, end = match.span()
        context_start = max(0, start - window)
        context_end = min(len(context), end + window)

        snippet = (
            context[context_start:start]
            + "["
            + context[start:end]
            + "]"
            + context[end:context_end]
        )
        print(f"Match {{i}}: {{snippet}}")

__mcp_find_regex_matches__()
del __mcp_find_regex_matches__
""".strip()
        )

        find_matches.emit()
        return find_matches.stdout

    @mcp.tool()
    async def get_context_metadata(ctx: Context):
        """
        This tool returns several useful metadata information about the context that might potentially be very long and therefore could not be loaded all at once.
        :return: A json with useful information about the context variable.
        """
        user_id = ctx.request_context.request.headers.get('x-user-id')
        chat_id = ctx.request_context.request.headers.get('x-chat-id')
        # Get the right repl manager
        repl_manager = SessionManager.get_instance().get(user_id, chat_id)

        length = repl_manager.execute(
            # language=Python
            """# noinspection PyUnresolvedReferences
if context:
    # noinspection PyUnresolvedReferences
    print(len(context))
            """.strip())
        length.emit()
        length = length.stdout
        context_type = repl_manager.execute(
            # language=Python
            """# noinspection PyUnresolvedReferences
if context:
    # noinspection PyUnresolvedReferences
    print(type(context))
            """.strip())
        context_type.emit()
        context_type = context_type.stdout

        return {
            'length': length,
            'type': context_type,
        }

    @mcp.tool()
    async def get_all_variable_names(ctx: Context):
        """
        This tool probes the repl environment and returns a list of all the variable names that are instantiated in there.
        :return: A list of variable names.
        """
        user_id = ctx.request_context.request.headers.get('x-user-id')
        chat_id = ctx.request_context.request.headers.get('x-chat-id')
        # Get the right repl manager
        repl_manager = SessionManager.get_instance().get(user_id, chat_id)
        var_names = repl_manager.execute(
            # language=Python
            """
def __mcp_emit_variable_names__():
    for name in list(globals().keys()):
        if '__' not in name:
            print(name)

__mcp_emit_variable_names__()
del __mcp_emit_variable_names__
        """.strip())
        var_names.emit()
        var_names = var_names.stdout
        return f"Here is a list of accessible variables in the repl environment: \n{var_names}" if len(var_names) else "There are no variables instantiated into the repl environment yet"


    @mcp.tool()
    async def peek_context(where_to_peek:Literal['head', 'middle', 'tail'], strategy: Literal['char', 'word', 'line'], amount: int, ctx: Context):
        """
        This tool returns a peek into the content of the context variable. It can return either the head, the middle or the tail of the context depending on the input parameter.
        :param where_to_peek: What portion of the text we want to peek into, can be either 'head', 'middle' or 'tail'.
        :param strategy: The type of item we want to peek, can be either 'char', 'word' or 'line'.
        :param amount: the amount of items we want to be returned by the tool.
        :return: A string with the items that were peeked into.
        """
        user_id = ctx.request_context.request.headers.get('x-user-id')
        chat_id = ctx.request_context.request.headers.get('x-chat-id')
        # Get the right repl manager
        repl_manager = SessionManager.get_instance().get(user_id, chat_id)


        peeked_items = repl_manager.execute(
            # language=Python
            f"""
def __mcp_print_context_slice__():
    if 'context' not in list(globals().keys()):
        print("No `context` variable found in REPL environment.")
    else:
        text = str(context)

        where = '{where_to_peek}'
        strat = '{strategy}'
        amount = {amount}

        if amount <= 0:
            print("")
        else:
            if strat == 'char':
                items = list(text)
                sep = ''
            elif strat == 'word':
                items = text.split()
                sep = ' '
            elif strat == 'line':
                items = text.splitlines()
                sep = '\\n'
            else:
                print("Unsupported strategy")
                items = None

            if items is not None:
                n = len(items)
                if n == 0:
                    print("")
                elif amount >= n:
                    print(sep.join(items))
                else:
                    if where == 'head':
                        sliced = items[:amount]
                    elif where == 'tail':
                        sliced = items[-amount:]
                    elif where == 'middle':
                        start = max(0, (n - amount) // 2)
                        end = start + amount
                        sliced = items[start:end]
                    else:
                        print("Unsupported where_to_peek")
                        sliced = None

                    if sliced is not None:
                        print(sep.join(sliced))
__mcp_print_context_slice__()
del __mcp_print_context_slice__
            """.strip())
        peeked_items.emit()
        return peeked_items.stdout
