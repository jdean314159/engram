# Writing a Custom Engine Adapter

If you need to use a backend that Engram doesn't support natively, implement
`LLMEngine` from `engram.engine.base`.

## Minimal implementation

```python
from engram.engine.base import LLMEngine

class MyEngine(LLMEngine):
    model_name = "my-model"
    is_cloud = False
    system_prompt = ""

    def generate(self, prompt: str, *, temperature: float = 0.7,
                 max_tokens: int = 2048, **kwargs) -> str:
        # Call your backend here
        response = my_backend.complete(prompt, temp=temperature, max_tok=max_tokens)
        return response.text
```

That's the minimum. `ProjectMemory` will work with this.

## Optional: logprob support

If your backend can return token log-probabilities, implement
`generate_with_logprobs()` to enable the `SurpriseFilter`:

```python
from engram.engine.base import LogprobResult, TokenLogprob

def generate_with_logprobs(self, prompt: str, **kwargs) -> LogprobResult:
    result = my_backend.complete_with_logprobs(prompt)
    token_logprobs = [
        TokenLogprob(token=t.text, logprob=t.logprob)
        for t in result.tokens
    ]
    return LogprobResult(text=result.text, token_logprobs=token_logprobs)
```

## Optional: token counting

The default `count_tokens()` uses `len(text) // 4` as an approximation.
Provide a real counter if your model has a tokenizer:

```python
def count_tokens(self, text: str) -> int:
    return len(self.tokenizer.encode(text))
```

## Optional: prompt compression

The pressure valve calls `compress_prompt()` when the prompt exceeds the
context window. The default raises `NotImplementedError`, which causes the
pressure valve to fall back to truncation instead.

```python
def compress_prompt(self, prompt: str, target_tokens: int) -> str:
    # Summarize or truncate to target_tokens
    return truncate_to_target(prompt, target_tokens)
```

## Registering with the failover router

```python
from engram.engine.router import FailoverEngine, FailoverPolicy

engine = FailoverEngine(
    engines=[MyEngine(), fallback_engine],
    policy=FailoverPolicy(max_attempts=3),
    name="my_failover",
)
```

Or use it directly as the `llm_engine` parameter to `ProjectMemory`.
