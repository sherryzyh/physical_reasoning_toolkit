# Model Clients Architecture

## Overview

The model clients package provides a unified interface for interacting with various AI model providers. All models inherit from `BaseModelClient` and handle image inputs according to their capabilities.

## Class Hierarchy

```
BaseModelClient (ABC)
├── OpenAIModel (supports vision - handles gpt-4.1, gpt-5xxxx, and o-family)
├── GeminiModel (vision support pending)
└── DeepseekModel (text-only)
```

## Design Principles

1. **BaseModelClient**: The abstract base class that defines the common interface for all model clients. Contains shared attributes (`model`, `provider`, `client`, `logger`) and the abstract `chat()` method.

2. **Unified Interface**: All models implement `chat(user_prompt: str, image_paths: Optional[List[str]] = None)`:
   - Models that support vision process images when provided
   - Models that don't support vision ignore images with a warning
   - All models support text-only prompts

3. **Concrete Implementations**: Provider-specific classes that inherit from `BaseModelClient` and implement the `chat()` method according to their capabilities.

4. **Factory Pattern**: The `create_model_client()` function automatically selects the appropriate client implementation based on the model name pattern.

## Supported Models

### OpenAI Models
- **gpt-4.1** and variants (gpt-4.1-mini, gpt-4.1-nano)
- **gpt-5xxxx** family (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
- **o-family** reasoning models (o3, o4, o4-mini, etc. - models starting with 'o' followed by a number)
- All OpenAI models support vision via the Responses API

### Google Gemini Models
- **gemini-*** (e.g., gemini-pro, gemini-1.5-pro)
- Vision support is planned but not yet implemented (images are ignored with a warning)

### DeepSeek Models
- **deepseek-*** (e.g., deepseek-chat)
- Text-only models (images are ignored with a warning)

## Usage

### Creating Model Clients

```python
from prkit.prkit_core.model_clients import create_model_client, BaseModelClient

# Using factory function (recommended)
client = create_model_client("gpt-5.1")

# Direct instantiation
from prkit.prkit_core.model_clients import OpenAIModel
client = OpenAIModel("gpt-5.1")
```

### Using Clients

```python
# Text-only call (all models)
prompt = "What is physics?"
response = client.chat(prompt)

# With images (models that support vision)
image_paths = [
    "https://example.com/image1.jpg",  # HTTP/HTTPS URL (used as-is)
    "/path/to/image2.png",              # File path (encoded to base64)
    "data:image/jpeg;base64,..."        # Base64 data URL (used as-is)
]
response = client.chat("What's in these images?", image_paths=image_paths)

# Models that don't support vision will warn and ignore images
deepseek_client = create_model_client("deepseek-chat")
response = deepseek_client.chat("Hello", image_paths=["image.jpg"])
# Warning: DeepSeek model does not support image inputs. Images will be ignored.
```

### OpenAI-Specific Features

```python
# O-family models automatically use reasoning with medium effort
o3_client = create_model_client("o3-mini")
response = o3_client.chat("Solve this physics problem step by step.")
```

## Image Input Handling

The `image_paths` parameter accepts three types of inputs:

1. **File paths**: Local file paths are automatically encoded to base64 data URLs
   ```python
   client.chat("Describe this image", image_paths=["/path/to/image.jpg"])
   ```

2. **HTTP/HTTPS URLs**: Remote URLs are passed directly to the API
   ```python
   client.chat("Describe this image", image_paths=["https://example.com/image.jpg"])
   ```

3. **Base64 data URLs**: Pre-encoded data URLs are used as-is
   ```python
   client.chat("Describe this image", image_paths=["data:image/jpeg;base64,..."])
   ```

## Adding New Providers

### Adding a New Model Provider

1. Create a new module (e.g., `anthropic.py`)
2. Create a class inheriting from `BaseModelClient`:

```python
from .base import BaseModelClient

class AnthropicModel(BaseModelClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        # Initialize provider-specific client
        self.provider = "anthropic"
        self.client = AnthropicClient(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def chat(self, user_prompt: str, image_paths: Optional[List[str]] = None):
        # Implement chat method
        # Handle image_paths according to provider capabilities
        if image_paths:
            self.logger.warning("Anthropic models don't support images yet")
        # ... implementation
        return response_text
```

3. Register in `factory.py`:

```python
from .anthropic import AnthropicModel

def create_model_client(model: str, logger=None) -> BaseModelClient:
    model_lower = model.lower()
    # ... existing code ...
    elif model_lower.startswith("claude"):
        return AnthropicModel(model, logger)
    # ...
```

4. Export in `__init__.py`:

```python
from .anthropic import AnthropicModel

__all__ = [
    # ... existing exports ...
    "AnthropicModel",
]
```


## Implementation Details

### OpenAI Implementation

- Uses OpenAI's **Responses API** (`client.responses.create()`)
- Supports vision via `input_image` content items
- O-family models automatically include `reasoning: {"effort": "medium"}`
- Returns `response.output_text` (SDK convenience property)

### Gemini Implementation

- Uses Google's `genai` SDK (`genai.Client`)
- Currently text-only (vision support planned)
- Supports additional kwargs for `GenerateContentConfig`

### DeepSeek Implementation

- Uses OpenAI-compatible API via `openai.OpenAI` with custom base URL
- Text-only (no vision support)
- Uses standard chat completions API

## Benefits of This Architecture

1. **Unified Interface**: Single `BaseModelClient` for all models simplifies the API
2. **Graceful Degradation**: Models that don't support vision warn and continue with text-only
3. **Type Safety**: ABC ensures all required methods are implemented
4. **Extensibility**: Easy to add new providers without modifying existing code
5. **Flexibility**: Each model handles images according to its capabilities
6. **Maintainability**: Simpler hierarchy makes the codebase easier to understand and maintain
7. **Future-Proof**: Easy to add vision support to existing models without breaking changes
