# Model Providers

PRKit includes a unified **model client** interface for running inference across multiple providers (LLMs and VLMs). The goal is to make it easy to switch models/providers while keeping the calling pattern consistent.

## Overview

- **Unified interface**: `BaseModelClient.chat(user_prompt: str, image_paths: Optional[List[str]] = None)`
- **Factory**: `create_model_client(model: str)` selects the right provider implementation based on the model name
- **Vision handling**: vision-capable providers consume `image_paths`; others ignore images with a warning

📖 For implementation details, see `src/prkit/prkit_core/model_clients/ARCHITECTURE.md`.

## Quick Start

```python
from prkit.prkit_core.model_clients import create_model_client

client = create_model_client("gpt-4.1-mini")
print(client.chat("State Newton's second law in one sentence."))
```

## Vision (Optional)

```python
from prkit.prkit_core.model_clients import create_model_client

client = create_model_client("gpt-4.1-mini")
text = client.chat(
    "Solve the problem shown in the image and return only the final answer.",
    image_paths=["/absolute/path/to/problem.png"],
)
print(text)
```

## Supported Providers

PRKit selects providers based on model name patterns (see `create_model_client()`):

| Provider | Example model names | Notes | Environment variables |
|---|---|---|---|
| OpenAI | `gpt-4.1-mini`, `gpt-5.1`, `o3-mini` | Vision supported via Responses API | `OPENAI_API_KEY` |
| Google Gemini | `gemini-pro`, `gemini-1.5-pro` | Text-only currently (images ignored with warning) | `GOOGLE_API_KEY` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | Text-only | `DEEPSEEK_API_KEY` |
| Ollama | `qwen3-vl`, `qwen3-vl:8b-instruct` | Local runtime; vision depends on model | (none) |

## Notes

- **Provider selection is model-driven**: you generally only specify a model string, not a provider name.
- **Image inputs**: pass absolute file paths, HTTP(S) URLs, or `data:image/...;base64,...` strings (see architecture doc for details).

