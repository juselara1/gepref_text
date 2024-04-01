from nltk.tokenize import word_tokenize, sent_tokenize
from gepref_text.base import AbstractTextStep

class WordTokenStep(AbstractTextStep):
    def __init__(self, lang: str="english"):
        self.lang = lang

    def call(self, data: str) -> str:
        tokens = word_tokenize(data, language=self.lang)
        return " ".join(tokens)

class SentTokenStep(AbstractTextStep):
    def __init__(self, lang: str="english"):
        self.lang = lang

    def call(self, data: str) -> str:
        tokens = sent_tokenize(data, language=self.lang)
        return " ".join(tokens)
