# Adapter Boundaries

## LLMEngine interface

All backends implement `LLMEngine` from `engram.engine.base`:

```python
class LLMEngine(ABC):
    model_name: str
    is_cloud: bool
    system_prompt: str
    supports_logprobs: bool = True

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.7,
                 max_tokens: int = 2048, **kwargs) -> str: ...

    def generate_with_logprobs(self, prompt: str, **kwargs) -> LogprobResult: ...
    def compress_prompt(self, prompt: str, target_tokens: int) -> str: ...
    def count_tokens(self, text: str) -> int: ...
```

`generate()` is the only required method. All others have defaults.

## SemanticLayerProtocol

Custom semantic backends implement:

```python
def search_generic_memories(
    self, query: str, *, limit: int = 20,
    per_type_limit: int = 60, include_graph: bool = True,
    graph_sentence_limit: int = 6,
) -> List[Dict[str, Any]]: ...
```

Each row should contain: `type`, `text`/`content`/`value`, `timestamp`,
`match_score`.

## Isolation guarantees

- Each `project_id` gets its own directory under `base_dir`. No data crosses
  between projects.
- Working memory uses a per-session named in-memory SQLite URI
  (`file:wm_{session_id}?mode=memory&cache=shared`), ensuring concurrent
  `ProjectMemory` instances don't share state.
- All `retrieve()` / `search()` calls filter by `project_id` in parameterised
  SQL to enforce isolation at the query level.
