"""
CS5760 NLP - Homework 1
Q1 Regex solutions + quick demo tests.

Run:
  python regex_tasks.py
"""

import re

REGEX = {
    # 1) U.S. ZIP codes: 12345, 12345-6789, 12345 6789. Whole tokens only.
    "zip": re.compile(r"\b\d{5}(?:[- ]\d{4})?\b"),

    # 2) Words that do NOT start with a capital letter.
    #    Allow internal apostrophes and hyphens: don't, state-of-the-art
    #    We'll treat a "word" as letters possibly followed by (' or -) + letters.
    "not_capital_start": re.compile(r"\b(?![A-Z])[A-Za-z]+(?:[\'’\-][A-Za-z]+)*\b"),

    # 3) Numbers with optional sign, thousands separators, decimals, and scientific notation.
    #    Examples: -1, +1, 1,234, 1,234.56, .5, 0.5, 1e10, 1.23E-4, -1,000.0e+2
    "number": re.compile(
        r"""
        (?<![\w.])                      # left boundary: avoid matching inside identifiers / decimals
        [+-]?                           # optional sign
        (?:                             # mantissa:
            (?:\d{1,3}(?:,\d{3})+|\d+)  #   digits with optional thousands commas OR plain digits
            (?:\.\d+)?                  #   optional decimal part
          | \.\d+                       # OR leading-decimal like .5
        )
        (?:[eE][+-]?\d+)?               # optional scientific notation
        (?![\w.])                       # right boundary
        """,
        re.VERBOSE,
    ),

    # 4) “email” spelling variants: email, e-mail, e mail (space, hyphen, en-dash), case-insensitive
    "email_variants": re.compile(r"\b e (?:[\s\-–])? mail \b", re.IGNORECASE | re.VERBOSE),

    # 5) go / goo / gooo ... as a word, optional trailing punctuation ! . , ?
    "go_interjection": re.compile(r"\bgo+\b[!.,?]?"),

    # 6) Lines ending with ? possibly followed only by closing quotes/brackets and spaces.
    #    Closing chars allowed: ) " ” ’ ] plus spaces.
    "line_ends_question_with_closers": re.compile(r"\?\s*[\)\"”’\]]*\s*$"),
}


def demo():
    print("Q1 demo tests\n" + "-" * 60)

    zip_text = "ZIPs: 12345 12345-6789 12345 6789; not 012345 or 123456."
    print("ZIP:", REGEX["zip"].findall(zip_text))

    cap_text = "Alice went to bob's state-of-the-art lab. don’t Stop. NASA uses rockets."
    print("Not starting with capital:", REGEX["not_capital_start"].findall(cap_text))

    num_text = "Nums: -1 +2 1,234 1,234.56 .5 0.5 1.23e-4 -1,000.0E+2 id123 not3.14x"
    print("Numbers:", REGEX["number"].findall(num_text))

    email_text = "Email me at EMAIL, e-mail, E mail, and also e–mail please."
    print("Email variants:", REGEX["email_variants"].findall(email_text))

    go_text = "go goo gooo gooooo! gooo? go, go. also gooooooo"
    print("Go interjection:", REGEX["go_interjection"].findall(go_text))

    lines = [
        'Is this a question?"   ',
        "Not a question.",
        "Really?)”  ",
        "Ends with question? ] ]  ",
    ]
    print("Line ends with question + closers:")
    for line in lines:
        m = REGEX["line_ends_question_with_closers"].search(line)
        print(f"  {line!r} -> {bool(m)}")


if __name__ == "__main__":
    demo()
