import os
from .openai_provider  import OpenAIProvider
from .claude_provider import ClaudeProvider

PROVIDERS = {
    'openai': OpenAIProvider,
    'claude': ClaudeProvider,
}


def get_provider():
    name = os.getenv('LLM_PROVIDER', 'openai').lower()
    provider_class = PROVIDERS.get(name)
    print("provider_class",provider_class)

    if not provider_class:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(PROVIDERS)}")
    return provider_class()
