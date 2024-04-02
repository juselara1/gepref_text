from abc import ABC, abstractmethod
from typing import Callable
from gepref_text.base import AbstractTextStep
from nltk.corpus import stopwords
from pydantic import PositiveInt

class AbstractTokenFilterStep(AbstractTextStep, ABC):

    def call(self, data: str) -> str:
        tokens = data.split(" ")
        filtered_tokens = filter(self.get_condition(), tokens)
        return " ".join(filtered_tokens)

    @abstractmethod
    def get_condition(self) -> Callable[[str], bool]:
        ...

class StopwordFilterStep(AbstractTokenFilterStep):
    def __init__(self, lang: str="english"):
        self.sw = stopwords.words(lang)

    def get_condition(self) -> Callable[[str], bool]:
        return lambda token: token not in self.sw

class TokenLenFilterStep(AbstractTokenFilterStep):
    def __init__(self, min_len: PositiveInt, max_len: PositiveInt):
        self.min_len = min_len
        self.max_len = max_len

    def get_condition(self) -> Callable[[str], bool]:
        return lambda token: (
                len(token) >= self.min_len and len(token) <= self.max_len
                )
