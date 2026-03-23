import os
import anthropic
from .base import BaseLLMProvider, LLMResponse

class ClaudeProvider(BaseLLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = model

    def complete(self, question: str, system_prompt: str) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,   # claude take system prompt separately
            messages=[
                {"role": 'user', 'content': question}
            ],

        )

        return LLMResponse(
            answer=response.content[0].text,
            model=self.model,
            provider='claude',
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )