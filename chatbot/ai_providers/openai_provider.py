import os
from openai import OpenAI
from .base import LLMResponse, BaseLLMProvider
from typing import List

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # use fine-tuned model if set, else default to gpt-4o
        self.model = model or os.getenv('OPENAI_MODEL') or 'gpt-4o'

        print(f'Using model: {self.model}')

    def complete(self, question: str, system_prompt: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content':system_prompt},
                {'role': 'user', 'content':question}
            ],
            temperature=0.7,
            max_tokens=500
        )
    
    def complete_with_messages(self, messages: List[dict]) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return LLMResponse(
            answer=response.choices[0].message.content,
            model=self.model,
            provider='openai',
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )