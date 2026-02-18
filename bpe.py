"""
CS5760 NLP - Homework 1
Q2: Mini-BPE learner for a toy corpus (character-level + end-of-word marker "_").

Run:
  python bpe.py --toy --merges 10
  python bpe.py --paragraph --merges 30
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import argparse
import re
from typing import List, Tuple, Dict


EOW = "_"


def word_to_symbols(word: str) -> List[str]:
    return list(word) + [EOW]


def get_bigram_counts(corpus: List[List[str]]) -> Counter[Tuple[str, str]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for toks in corpus:
        for a, b in zip(toks, toks[1:]):
            counts[(a, b)] += 1
    return counts


def merge_pair_in_word(tokens: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
    a, b = pair
    out = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(new_token)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def merge_pair_in_corpus(corpus: List[List[str]], pair: Tuple[str, str], new_token: str) -> List[List[str]]:
    return [merge_pair_in_word(toks, pair, new_token) for toks in corpus]


def initial_vocab(corpus: List[List[str]]) -> set[str]:
    v = set()
    for toks in corpus:
        v.update(toks)
    return v


@dataclass
class BPEResult:
    merges: List[Tuple[Tuple[str, str], str]]  # ((a,b), "ab")
    vocab: set[str]


def learn_bpe(corpus_words: List[str], num_merges: int = 10) -> BPEResult:
    corpus = [word_to_symbols(w) for w in corpus_words]
    vocab = initial_vocab(corpus)
    merges: List[Tuple[Tuple[str, str], str]] = []

    for step in range(1, num_merges + 1):
        counts = get_bigram_counts(corpus)
        if not counts:
            break

        best_count = counts.most_common(1)[0][1]
        # deterministic tie-break: max count, then lexicographically smallest pair
        best_pairs = [p for p, c in counts.items() if c == best_count]
        best_pair = sorted(best_pairs)[0]
        new_token = "".join(best_pair)

        print(f"Step {step:02d}: best pair {best_pair} -> '{new_token}'  (count={best_count})")

        corpus = merge_pair_in_corpus(corpus, best_pair, new_token)
        vocab.add(new_token)
        merges.append((best_pair, new_token))
        print(f"         vocab size now: {len(vocab)}")

        # stop if no pair repeats
        next_counts = get_bigram_counts(corpus)
        if next_counts and next_counts.most_common(1)[0][1] < 2:
            break

    return BPEResult(merges=merges, vocab=vocab)


def apply_bpe(word: str, merges: List[Tuple[Tuple[str, str], str]]) -> List[str]:
    tokens = word_to_symbols(word)
    for (a, b), new_tok in merges:
        tokens = merge_pair_in_word(tokens, (a, b), new_tok)
    return tokens


def toy_corpus_words() -> List[str]:
    return (
        ["low"] * 5
        + ["lowest"] * 2
        + ["newer"] * 6
        + ["wider"] * 3
        + ["new"] * 2
    )


def paragraph_text() -> str:
    # 4–6 sentences (English example; you can replace with your own language).
    return (
        "Subword tokenization is a practical compromise between word and character models. "
        "It reduces out-of-vocabulary problems by breaking rare words into frequent pieces. "
        "Byte Pair Encoding learns merges that often align with stems and suffixes in English. "
        "However, merges are data-driven and may split words in unexpected ways. "
        "Careful preprocessing and evaluation are important for downstream tasks."
    )


def paragraph_words(text: str) -> List[str]:
    # simple word extraction for training merges (letters only)
    return re.findall(r"[A-Za-z]+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="Train on the toy corpus from class.")
    ap.add_argument("--paragraph", action="store_true", help="Train on the included short paragraph.")
    ap.add_argument("--merges", type=int, default=10, help="Number of merges to learn.")
    args = ap.parse_args()

    if args.toy:
        words = toy_corpus_words()
        print("Training on toy corpus:", " ".join(words))
    elif args.paragraph:
        text = paragraph_text()
        words = paragraph_words(text)
        print("Training on paragraph text:\n", text, "\n")
        print("Training words:", words)
    else:
        raise SystemExit("Choose --toy or --paragraph")

    result = learn_bpe(words, num_merges=args.merges)

    # Segment required words for Q2.2
    if args.toy:
        test_words = ["new", "newer", "lowest", "widest", "newestest"]
        print("\nSegmentation (toy merges):")
        for w in test_words:
            print(f"  {w:10s} -> {apply_bpe(w, result.merges)}")

    if args.paragraph:
        # report top 5 merges and 5 longest tokens
        print("\nTop 5 merges learned:")
        for i, (pair, newtok) in enumerate(result.merges[:5], start=1):
            print(f"  {i}. {pair} -> {newtok}")

        longest = sorted(result.vocab, key=len, reverse=True)[:5]
        print("\n5 longest subword tokens:")
        for t in longest:
            print(" ", t)

        # segment 5 words from the paragraph
        sample = ["tokenization", "practical", "unexpected", "downstream", "evaluation"]
        print("\nSegmentation (paragraph merges):")
        for w in sample:
            print(f"  {w:12s} -> {apply_bpe(w.lower(), result.merges)}")


if __name__ == "__main__":
    main()
