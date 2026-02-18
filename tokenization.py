"""
CS5760 NLP - Homework 1
Q5: Tokenization comparison (naive vs manual vs tool tokenizer).

Run:
  python tokenization.py
"""

import re
from nltk.tokenize import wordpunct_tokenize


PARAGRAPH = (
    "Yesterday, I visited New York City and tried state-of-the-art espresso. "
    "I can't believe how fast the subway is—it's amazing! "
    "Later, I met my friend at Central Park, and we talked about machine learning."
)

MWES = [
    "New York City",
    "Central Park",
    "machine learning",
]


def naive_space_tokenize(text: str):
    return text.split()


def manual_tokenize(text: str):
    # Separate punctuation while preserving contractions/hyphenated words.
    # 1) Normalize em/en dashes to space-surrounded token
    text = re.sub(r"[—–]", " — ", text)
    # 2) Split punctuation (.,!?;:) but keep apostrophes inside words (can't) and hyphens in compounds.
    tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?(?:-[A-Za-z]+)*|[0-9]+|—|[.,!?;:]", text)
    return tokens


def tool_tokenize(text: str):
    # NLTK wordpunct_tokenize does not need external models/downloads.
    return wordpunct_tokenize(text)


def main():
    print("Paragraph:\n", PARAGRAPH, "\n")

    naive = naive_space_tokenize(PARAGRAPH)
    manual = manual_tokenize(PARAGRAPH)
    tool = tool_tokenize(PARAGRAPH)

    print("Naive space tokens:\n", naive, "\n")
    print("Manual corrected tokens:\n", manual, "\n")
    print("Tool tokens (NLTK wordpunct_tokenize):\n", tool, "\n")

    # Differences
    naive_set = set(naive)
    manual_set = set(manual)
    tool_set = set(tool)

    print("Tokens in manual but not naive:", sorted(manual_set - naive_set))
    print("Tokens in tool but not manual:", sorted(tool_set - manual_set))

    print("\nMWEs identified:")
    for m in MWES:
        print(" -", m)

    print("\nWhy MWEs should be single tokens (short notes):")
    print(" - They refer to a single named entity or fixed phrase, so splitting can hurt meaning.")
    print(" - Many NLP models treat them as a unit (e.g., NER, phrase embeddings).")
    print(" - They often behave like one semantic concept in classification and retrieval.")


if __name__ == "__main__":
    main()
