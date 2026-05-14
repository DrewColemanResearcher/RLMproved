import asyncio
import sys
from mcp.server.fastmcp import FastMCP, Context
from starlette.requests import Request
from starlette.responses import JSONResponse

from rlm_mcp.rlm_mcp_server.stateful_mcp.session_manager import SessionManager
from rlm_mcp.rlm_mcp_server.tools.probing_tools import register as register_probing_tools
from rlm_mcp.rlm_mcp_server.tools.execution_tools import register as register_execution_tools

# We want to suppress some stupid warnings
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

mcp = FastMCP("RLM")

register_probing_tools(mcp)
register_execution_tools(mcp)


@mcp.custom_route(
    "/internal/repl/context",
    methods=["POST"],
    include_in_schema=False,
)
async def preload_repl_context(request: Request) -> JSONResponse:
    user_id = request.headers.get("x-user-id")
    chat_id = request.headers.get("x-chat-id")

    if not user_id or not chat_id:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "Missing x-user-id or x-chat-id header."},
        )

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "Invalid JSON body."},
        )

    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, str):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "Body must include a string 'context'."},
        )

    repl_manager = SessionManager.get_instance().get(user_id, chat_id)
    context = context.strip()
    context = context.encode('utf-8').decode('ascii', errors='ignore')
    preload_result = repl_manager.execute(f"context = {context!r}")
    if preload_result.stderr:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "Failed to set context in REPL environment.",
                "details": preload_result.stderr,
            },
        )

    return JSONResponse(
        content={
            "ok": True,
            "user_id": user_id,
            "chat_id": chat_id,
            "context_length": len(context),
        }
    )

if __name__ == '__main__':
    mcp.run('streamable-http')
