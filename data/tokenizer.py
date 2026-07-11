from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges: List[List[str]] = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            pair_counts = Counter(
                (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)
            )

            # Prefer higher count; break ties lexicographically
            best_pair = min(pair_counts, key=lambda p: (-pair_counts[p], p))
            merges.append([best_pair[0], best_pair[1]])

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges
