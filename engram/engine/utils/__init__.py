"""Engine utilities — structured output parsing."""

__all__ = ['StructuredOutputHandler', 'StructuredOutputError', 'ParseResult']

def __getattr__(name):
    if name in __all__:
        from .structured_output import StructuredOutputHandler, StructuredOutputError, ParseResult
        return {'StructuredOutputHandler': StructuredOutputHandler,
                'StructuredOutputError': StructuredOutputError,
                'ParseResult': ParseResult}[name]
    raise AttributeError(f"module 'engram.engine.utils' has no attribute {name!r}")
