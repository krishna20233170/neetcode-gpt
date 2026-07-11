from typing import List, Dict, Iterator


class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._tokenize(str(value), vocab) for value in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return sum(1 for _ in self._scan_tokens(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        word_count = len(text.split())
        token_count = self.count_tokens(text, vocab)
        return round(token_count / word_count, 4)

    def _tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        return list(self._scan_tokens(text, vocab))

    def _scan_tokens(self, text: str, vocab: Dict[str, int]) -> Iterator[str]:
        index = 0
        n = len(text)

        while index < n:
            match = text[index]
            for end in range(n, index, -1):
                piece = text[index:end]
                if piece in vocab:
                    match = piece
                    break
            yield match
            index += len(match)
