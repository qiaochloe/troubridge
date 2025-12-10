"""
Utility for training a HuggingFace tokenizer with fixed delimiter tokens.

The tokenizer is built with the `tokenizers` library (WordLevel model) and
ensures the following tokens are always present in the vocabulary:
["::", ",", "(", ")", "\\n"].
"""

from json import load
from pathlib import Path
from typing import Iterable, Optional

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordLevelTrainer

# Tokens we must always retain in the vocabulary.
DELIMITER_TOKENS = ["::", ",", "(", ")", "\n"]
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


def _stack_traces(data_path: str) -> Iterable[str]:
    """Yield stack trace strings from the hotness dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = load(f)
    for item in data:
        yield item["stack_trace"]


def build_tokenizer(data_path: str = "data/hotness_data.json", vocab_size: int = 5000) -> Tokenizer:
    """
    Train and return a HuggingFace `Tokenizer` configured to keep delimiter tokens.

    The tokenizer uses a WordLevel model with a regex pre-tokenizer that isolates
    our delimiter tokens so they appear as standalone tokens.
    """
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Split(pattern=r"(::|,|\(|\)|\n)", behavior="isolated")

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=[UNK_TOKEN, PAD_TOKEN] + DELIMITER_TOKENS,
    )

    tokenizer.train_from_iterator(_stack_traces(data_path), trainer=trainer)
    return tokenizer


class TokenizerVocab:
    """
    Light wrapper to mimic the old `vocab` interface used by the model.

    - `len(wrapper)` returns the vocabulary size.
    - `wrapper[token]` returns the token id, falling back to unk if missing.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._unk_id = tokenizer.token_to_id(UNK_TOKEN)

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __getitem__(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return token_id if token_id is not None else self._unk_id


def create_vocab(data_path: str = "data/hotness_data.json", vocab_size: int = 5000) -> TokenizerVocab:
    """
    Backwards-compatible entry point that returns a vocab-like wrapper
    backed by the HuggingFace tokenizer.
    """
    tokenizer = build_tokenizer(data_path=data_path, vocab_size=vocab_size)
    return TokenizerVocab(tokenizer)


def tokenize(trace: str, tokenizer: Optional[Tokenizer] = None) -> list[str]:
    """
    Tokenize a stack trace into a list of tokens using the HuggingFace tokenizer.
    """
    tokenizer = tokenizer or build_tokenizer()
    return tokenizer.encode(trace).tokens


def save_tokenizer(tokenizer: Tokenizer, path: str = "tokenizer.json") -> str:
    """
    Persist the tokenizer to disk for reuse in training/inference pipelines.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return str(output_path)


if __name__ == "__main__":
    tk = build_tokenizer()
    save_path = save_tokenizer(tk, "tokenizer.json")
    print(f"Tokenizer saved to {save_path}")