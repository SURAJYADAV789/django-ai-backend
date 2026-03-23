from abc import ABC, abstractmethod
from dataclasses import dataclass


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

