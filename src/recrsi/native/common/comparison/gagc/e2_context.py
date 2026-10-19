"""Bind native Deep Agents compaction to E2's actual request envelope.

The model catalogue's 1.05M context is not our128k run limit. Reuse native
summarization/history retrieval; do not drop task inputs or rewrite candidates.
"""
from pathlib import Path


def context_middleware(model, logs_root):
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from se_shared_api_ledger import _estimate_tokens

    def wire_tokens(messages, *, tools=None):
        payload = model._get_request_payload(messages,
            tools=[convert_to_openai_tool(tool) for tool in tools] if tools else [])
        return _estimate_tokens(payload)

    # Only the middleware gets this writable backend. The controller retains
    # its read-only task/log backend and may read the summary's /logs/ pointer.
    history = CompositeBackend(default=StateBackend(), artifacts_root='/logs',
        routes={'/logs/': FilesystemBackend(root_dir=str(Path(logs_root).resolve()), virtual_mode=True)})
    return SummarizationMiddleware(model=model, backend=history,
        trigger=('tokens', 100000), keep=('messages', 6), token_counter=wire_tokens,
        trim_tokens_to_summarize=None, truncate_args_settings=None)
