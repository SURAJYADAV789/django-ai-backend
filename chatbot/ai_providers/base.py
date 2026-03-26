from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class LLMResponse:
    answer : str
    model : str
    provider : str
    input_tokens : str
    output_tokens : str

class BaseLLMProvider(ABC):

    @abstractmethod
    def complete(self, question: str, system_prompt: str) -> LLMResponse:
        pass

    @abstractmethod
    def complete_with_messages(self, messages: List[dict]) -> LLMResponse:
        '''Used for the conversation - receives pre-built messages list'''
        pass
