# LLM/VLM Providers

Unified interface for integrating Large Language Models (LLMs) and Vision-Language Models (VLMs) with the Physical Reasoning Toolkit.

## Overview

The toolkit provides a unified `LLMClient` interface that abstracts away provider-specific differences, allowing you to easily switch between different LLM/VLM providers. Currently supported providers include OpenAI, Google Gemini, and DeepSeek, with more providers coming soon.

## Quick Start

```python
from prkit.prkit_core.llm import LLMClient

# Automatically create the right client based on model name
client = LLMClient.from_model("gpt-4o")

# Send a chat message
messages = [
    {"role": "system", "content": "You are a physics expert."},
    {"role": "user", "content": "What is Newton's second law?"}
]
response = client.chat(messages)
print(response)
```

## Supported Providers

| Provider | Models | Status | API Key Environment Variable |
|----------|--------|--------|------------------------------|
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5, Reasoning API (o1, o3) | ✅ Available | `OPENAI_API_KEY` |
| **Google Gemini** | gemini-pro, gemini-1.5-pro, etc. | ✅ Available | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| **DeepSeek** | deepseek-chat, deepseek-reasoner, etc. | ✅ Available | `DEEPSEEK_API_KEY` |
| **More Providers** | Coming soon | 🔜 Planned | - |

## Provider Details

### OpenAI

OpenAI provides multiple model families through different APIs:

#### Chat Models (GPT-4o, GPT-4, GPT-3.5)

**Models:**
- `gpt-4o`: Latest multimodal model with vision capabilities
- `gpt-4`: Advanced reasoning model
- `gpt-3.5-turbo`: Fast and cost-effective model

**Features:**
- Standard chat completions
- Structured output support (GPT-4o only)
- Temperature control

**Usage:**
```python
from prkit.prkit_core.llm import LLMClient

# Create OpenAI chat client
client = LLMClient.from_model("gpt-4o")

# Basic chat
messages = [
    {"role": "user", "content": "Explain quantum mechanics"}
]
response = client.chat(messages)

# Structured output (GPT-4o only)
response = client.chat_structured(
    system_prompt="You are a physics tutor.",
    prompt="List the three laws of thermodynamics.",
    response_format="json"
)
```

**Environment Setup:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Reasoning API Models (o1, o3)

**Models:**
- `o1-preview`, `o1-mini`: Advanced reasoning models
- `o3-mini`: Latest reasoning model

**Features:**
- Enhanced reasoning capabilities
- Structured output support
- Optimized for complex problem-solving

**Usage:**
```python
# Create OpenAI reasoning client
client = LLMClient.from_model("o1-preview")

# The reasoning API handles complex reasoning automatically
messages = [
    {"role": "user", "content": "Solve this physics problem: [problem statement]"}
]
response = client.chat(messages)
```

**Note:** Reasoning API models automatically use the `OAIReasonModel` class, which uses the `responses.create` endpoint.

### Google Gemini

Google's Gemini models provide strong multimodal capabilities and reasoning.

**Models:**
- `gemini-pro`: General-purpose model
- `gemini-1.5-pro`: Advanced model with extended context
- `gemini-1.5-flash`: Fast and efficient model

**Features:**
- Multimodal support (text, images)
- System instructions
- Flexible configuration

**Usage:**
```python
from prkit.prkit_core.llm import LLMClient

# Create Gemini client
client = LLMClient.from_model("gemini-pro")

# Basic chat
messages = [
    {"role": "system", "content": "You are a physics expert."},
    {"role": "user", "content": "Explain the photoelectric effect"}
]
response = client.chat(messages)

# With additional configuration
response = client.chat(
    messages,
    temperature=0.7,
    max_output_tokens=2048
)
```

**Environment Setup:**
```bash
# Either variable works (GEMINI_API_KEY is preferred)
export GEMINI_API_KEY="your-gemini-api-key"
# OR
export GOOGLE_API_KEY="your-google-api-key"
```

**Note:** Gemini models use the Google GenAI SDK and automatically convert OpenAI-style messages to Gemini format.

### DeepSeek

DeepSeek provides cost-effective models with strong reasoning capabilities.

**Models:**
- `deepseek-chat`: General-purpose chat model
- `deepseek-reasoner`: Advanced reasoning model

**Features:**
- OpenAI-compatible API
- Cost-effective pricing
- Strong reasoning capabilities

**Usage:**
```python
from prkit.prkit_core.llm import LLMClient

# Create DeepSeek client
client = LLMClient.from_model("deepseek-chat")

# Basic chat
messages = [
    {"role": "user", "content": "What is the Schrödinger equation?"}
]
response = client.chat(messages)
```

**Environment Setup:**
```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

**Note:** DeepSeek uses an OpenAI-compatible API, so the interface is similar to OpenAI models.

## Unified Interface

### LLMClient Base Class

All provider implementations inherit from `LLMClient` and provide a consistent interface:

```python
class LLMClient:
    def __init__(self, model: str, logger=None):
        """Initialize LLM client."""
        
    def chat(self, messages):
        """Send chat messages to the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response text from the model
        """
        
    def chat_structured(self, system_prompt, prompt, response_format=None):
        """Send structured chat request with format specification.
        
        Args:
            system_prompt: System instruction prompt
            prompt: User prompt
            response_format: Expected response format specification
            
        Returns:
            Parsed structured response
        """
```

### Automatic Provider Detection

The `from_model()` static method automatically creates the appropriate client:

```python
# Automatically detects provider from model name
client = LLMClient.from_model("gpt-4o")        # OpenAI
client = LLMClient.from_model("gemini-pro")    # Google Gemini
client = LLMClient.from_model("deepseek-chat") # DeepSeek
```

**Model Name Patterns:**
- Models starting with `gpt` → OpenAI Chat API
- Models starting with `o` → OpenAI Reasoning API
- Models containing `deepseek` → DeepSeek
- Models starting with `gemini` → Google Gemini

## Usage Examples

### Basic Chat

```python
from prkit.prkit_core.llm import LLMClient

# Create client
client = LLMClient.from_model("gpt-4o")

# Simple conversation
messages = [
    {"role": "user", "content": "What is Newton's first law?"}
]
response = client.chat(messages)
print(response)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a physics tutor."},
    {"role": "user", "content": "What is momentum?"},
]

# First response
response1 = client.chat(messages)
print(response1)

# Continue conversation
messages.append({"role": "assistant", "content": response1})
messages.append({"role": "user", "content": "How is it conserved?"})
response2 = client.chat(messages)
print(response2)
```

### Structured Output (OpenAI GPT-4o)

```python
client = LLMClient.from_model("gpt-4o")

response = client.chat_structured(
    system_prompt="You are a physics problem analyzer.",
    prompt="Analyze this problem: A ball is thrown upward with velocity 10 m/s.",
    response_format={
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "concepts": {"type": "array", "items": {"type": "string"}},
            "difficulty": {"type": "string"}
        }
    }
)
print(response)  # Parsed JSON object
```

### Switching Between Providers

```python
# Easy to switch providers for comparison
models = ["gpt-4o", "gemini-pro", "deepseek-chat"]
prompt = "Explain the double-slit experiment"

for model_name in models:
    client = LLMClient.from_model(model_name)
    response = client.chat([
        {"role": "user", "content": prompt}
    ])
    print(f"{model_name}: {response[:100]}...")
```

### Using with Physical Reasoning Tasks

```python
from prkit.prkit_core.llm import LLMClient
from prkit.prkit_datasets import DatasetHub

# Load a physics problem
hub = DatasetHub()
dataset = hub.load("phybench", variant="full", auto_download=True)
problem = dataset[0]

# Create LLM client
client = LLMClient.from_model("gpt-4o")

# Solve the problem
messages = [
    {"role": "system", "content": "You are a physics problem solver. Provide step-by-step solutions."},
    {"role": "user", "content": f"Problem: {problem.question}\n\nSolve this step by step."}
]
solution = client.chat(messages)
print(solution)
```

## Provider-Specific Notes

### OpenAI

- **Structured Output**: Only available for GPT-4o using `chat_structured()`
- **Reasoning API**: Uses different endpoint (`responses.create`) with automatic reasoning
- **Temperature**: Defaults to 0 for chat models (deterministic)

### Google Gemini

- **Message Format**: Automatically converts OpenAI-style messages to Gemini format
- **System Instructions**: Supported via system role in messages
- **Multimodal**: Can handle images (future enhancement)
- **API Key**: Supports both `GEMINI_API_KEY` and `GOOGLE_API_KEY` for backward compatibility

### DeepSeek

- **API Compatibility**: Uses OpenAI-compatible API, so interface is identical
- **Base URL**: Automatically set to `https://api.deepseek.com`
- **Cost**: Generally more cost-effective than OpenAI

## Error Handling

```python
from prkit.prkit_core.llm import LLMClient

try:
    client = LLMClient.from_model("gpt-4o")
    response = client.chat(messages)
except ValueError as e:
    print(f"Unknown model: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Logging

All LLM clients use the centralized logging system:

```python
from prkit.prkit_core import PRKitLogger

# Logging is automatically configured
logger = PRKitLogger.get_logger(__name__)
client = LLMClient.from_model("gpt-4o", logger=logger)
```

## Environment Variables

Set up your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Google Gemini (either works)
export GEMINI_API_KEY="..."
# OR
export GOOGLE_API_KEY="..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
```

Or use a `.env` file (automatically loaded):

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...
```

## Future Providers

The following providers are planned for future releases:

- **Anthropic Claude**: Claude 3 models
- **Meta Llama**: Llama 3 and future models
- **Mistral AI**: Mistral models
- **More**: Additional providers based on community needs

## Contributing New Providers

To add a new provider:

1. Create a new class inheriting from `LLMClient`
2. Implement `chat()` method (and optionally `chat_structured()`)
3. Update `from_model()` to recognize the new provider's model names
4. Add provider documentation to this file
5. Add tests for the new provider

See existing implementations in `src/prkit/prkit_core/llm.py` for examples.

## Best Practices

1. **Use `from_model()`**: Let the toolkit automatically detect the provider
2. **Handle Errors**: Wrap API calls in try-except blocks
3. **Set API Keys**: Use environment variables for security
4. **Logging**: Use the provided logger for debugging
5. **Structured Output**: Use `chat_structured()` when you need JSON responses (OpenAI GPT-4o)
6. **Provider Comparison**: Easy to compare responses across providers using the unified interface

## Troubleshooting

### API Key Not Found

```python
# Check environment variables
import os
print(os.environ.get("OPENAI_API_KEY"))  # Should not be None
```

### Unknown Model Error

```python
# Make sure model name matches expected patterns
# OpenAI: gpt-4o, gpt-4, gpt-3.5-turbo, o1-preview
# Gemini: gemini-pro, gemini-1.5-pro
# DeepSeek: deepseek-chat, deepseek-reasoner
```

### Structured Output Not Supported

```python
# Only GPT-4o supports structured output currently
# For other models, parse JSON manually from chat() response
```
