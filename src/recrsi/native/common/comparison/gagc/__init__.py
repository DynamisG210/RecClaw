# Lazy imports — agent.py requires langchain/langgraph at runtime
__all__ = ["create_gagc_agent"]


def create_gagc_agent(*args, **kwargs):
    from gagc.agent import create_gagc_agent as _impl  # noqa: PLC0415

    return _impl(*args, **kwargs)
