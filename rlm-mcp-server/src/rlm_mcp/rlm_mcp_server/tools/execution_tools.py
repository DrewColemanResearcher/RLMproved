import os

from mcp.server.fastmcp import Context
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rlm_mcp.rlm_mcp_server.stateful_mcp.session_manager import SessionManager


def register(mcp):
    @mcp.tool()
    async def run_code(code: str, ctx: Context):
        """
        A tool for running arbitrary code in the repl environment.
        For any effect to be seen, a print statement must be used.
            Examples:
                context[:500] will not return anything.
                print(context[:500]) will print the first 500 characters of the context.
                a = 3 will not return anything.
                print(a) will return the value of a if a was previously defined.
        :param code: The python code to run.
        :return: A ReplResponse containing the stdout and the stderr of the execution. Moreover, there is a returncode just for completeness.
                 The value returned by this tool will always be truncated to 100_000 characters.
        """
        user_id = ctx.request_context.request.headers.get('x-user-id')
        chat_id = ctx.request_context.request.headers.get('x-chat-id')
        print(f"==Running CUSTOM CODE for chat {chat_id}==")
        print(code)
        print("==============")
        # Get the right repl manager
        repl_manager = SessionManager.get_instance().get(user_id, chat_id)
        execution_result = repl_manager.execute(code)
        execution_result.emit()
        execution_result.stdout = execution_result.stdout[:100_000]
        execution_result.stderr = execution_result.stderr[:100_000]
        return execution_result


    @mcp.tool()
    async def sub_llm(prompt: str, ctx: Context):
        """
        This tool lets you ask an LLM anything. This is supposed to be used for language processing tasks. Avoid passing overly-long prompts to this function.
        The LLM is not aware of your previous conversations nor of the content of the variables. It must be used in a single shot and provided with everything it might need in a single prompt.
        Negative Example:
            prompt = "Summarize everything I have told you so far in a concise way." -> Error!!! The model does not know what it was told.
        Positive Example:
            prompt = "Summarize this passage:\n When shall we three meet again In thunder, lightning, or in rain?" -> Correct! The model will say something like "The passage is a famous opening from Macbeth by William Shakespeare".
        Could also be used to plan what to do to solve a problem by asking for a strategy.
        :param prompt: The prompt to be sent to the LLM.
        :return: A response from an LLM.
        """
        model_name = ctx.request_context.request.headers.get("x-sub-llm-model")
        api_key = (
                ctx.request_context.request.headers.get("x-openrouter-api-key")
                or os.getenv("OPENROUTER_API_KEY")
        )

        if not model_name:
            return "Missing header: x-sub-llm-model"

        if not api_key:
            return "Missing OpenRouter API key"

        try:
            provider = OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            model = OpenAIChatModel(
                model_name,
                provider=provider,
            )
            agent = Agent(
                model,
                instructions="Return plain text only. Be concise but thorough. Only answer to the question, do not prompt the user for additional help or information. Do not suggest follow up questions.",
            )
            result = await agent.run(prompt)
            print(f"SUB_LLM_CALLED {model_name}, result: {result}")
            return result.output

        except Exception as e:
            return f"sub_llm failed: {type(e).__name__}: {e}"
