# CS5760 Natural Language Processing — Homework 1 (Spring 2026)

**Student:** DivyaBaireddy  
**Course:** CS5760 NLP  
**University:** University of Central Missouri  

> Replace the student name/ID above if needed.

---

## Submission (GitHub)
1. Create a GitHub repo (example name: `cs5760-hw1`).
2. Upload these files and commit:
   - `regex_tasks.py`
   - `bpe.py`
   - `tokenization.py`
   - `requirements.txt`
   - `README.md`
3. Your submission link (paste this on Brightspace):  
   **GitHub Repo:** `https://github.com/<divyabaireddy>/cs5760-hw1`

---

# Q1. Regex

All patterns are implemented in `regex_tasks.py` (run it to see demo matches).

### 1) U.S. ZIP codes
- Matches: `12345`, `12345-6789`, `12345 6789`
- Whole token only:
```regex
\b\d{5}(?:[- ]\d{4})?\b
```

### 2) Words that do **not** start with a capital letter
- Allows internal apostrophes/hyphens: `don't`, `state-of-the-art`
```regex
\b(?![A-Z])[A-Za-z]+(?:[\'’\-][A-Za-z]+)*\b
```

### 3) Numbers (sign, commas, decimals, scientific notation)
Covers examples like: `-1`, `+2`, `1,234`, `1,234.56`, `.5`, `0.5`, `1.23e-4`
(see the verbose pattern in `regex_tasks.py`).

### 4) “email” spelling variants (case-insensitive)
Matches `email`, `e-mail`, `e mail`, and also `e–mail` (en dash):
```regex
\b e (?:[\s\-–])? mail \b
```

### 5) go / goo / gooo … (optional trailing punctuation)
```regex
\bgo+\b[!.,?]?
```

### 6) Lines ending with `?` with optional closing quotes/brackets
Matches a line that ends with `?` and then only closers/spaces:
```regex
\?\s*[\)\"”’\]]*\s*$
```

---

# Q2. Manual BPE on a toy corpus

Toy corpus (from class):
```
low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new
```

## 2.1 Add end-of-word `_` and initial vocabulary
Example tokenization:
- `low_` → `l o w _`
- `lowest_` → `l o w e s t _`

**Initial vocabulary (characters + `_`):**  
`{ l, o, w, e, s, t, n, r, i, d, _ }`

## Bigram counts + first 3 merges (by hand)

### Step 1 (most frequent pair)
Most frequent pair count is **9** (tie between `e r` and `r _`).  
I choose to merge **`e r → er`**.

Updated snippet (showing 2 lines):
- `newer_` becomes `n e w er _`
- `wider_` becomes `w i d er _`

**New token:** `er`  
**Updated vocabulary:** previous + `er`

### Step 2
Most frequent pair is now `er _` with count **9** → merge:
- **`er _ → er_`**

Snippet:
- `newer_` becomes `n e w er_`
- `wider_` becomes `w i d er_`

**New token:** `er_`  
**Updated vocabulary:** previous + `er_`

### Step 3
Most frequent pair count is **8** (tie between `n e` and `e w`).  
I choose to merge **`n e → ne`**.

Snippet:
- `newer_` becomes `ne w er_`
- `new_` becomes `ne w _`

**New token:** `ne`  
**Updated vocabulary:** previous + `ne`

## 2.2 Code a mini-BPE learner
Implemented in `bpe.py`.

Run:
```bash
python bpe.py --toy --merges 10
```

Required segmentations (printed by the script):
- `new`
- `newer`
- `lowest`
- `widest`
- invented word: `newestest`

### Explanation (5–6 sentences)
Subword tokens reduce OOV issues because a model can build unseen words from known pieces.  
For example, even if `widest` never appears in training, the tokenizer can still represent it using learned subwords like `w`, `i`, `d`, and common suffix pieces if learned.  
This helps downstream models generalize by reusing frequent components across many words.  
In English, BPE often learns meaningful morpheme-like units such as `er_` in comparative forms (`newer_`, `wider_`).  
However, merges are driven by frequency, so sometimes subwords do not align perfectly with linguistic boundaries.  
Even then, subword segmentation usually improves robustness compared with word-only vocabularies.

## 2.3 Train BPE on a short paragraph (English example)
Implemented in `bpe.py` with a 5-sentence paragraph.

Run:
```bash
python bpe.py --paragraph --merges 30
```

The script prints:
- the **five most frequent merges**
- the **five longest subword tokens**
- segmentation of 5 words from the paragraph

### Reflection (5–8 sentences)
With a small paragraph, BPE tends to learn high-frequency stems and common suffixes (like `tion_`, `ing_`, etc.) if they repeat.  
It may also learn whole words when the corpus is tiny and repetition is high.  
Pros: it reduces OOV and makes the model handle rare/derived forms more smoothly.  
Pros: it provides a compact vocabulary compared with full word-level tokenization.  
Cons: merges can be unstable on small datasets and may overfit to accidental repetitions.  
Cons: segmentation may not respect morphology for some words, causing awkward splits.  
Overall, subwords are a good compromise but must be trained on sufficiently representative text.

---

# Q3. Bayes Rule Applied to Text
- **P(c)**: prior probability of class `c` (how common the class is before seeing the document).
- **P(d|c)**: likelihood of the document `d` assuming the class is `c` (how probable the words/features are under that class).
- **P(c|d)**: posterior probability of class `c` after observing document `d` (what we want for classification).

When comparing classes, the denominator **P(d)** can be ignored because it is the same for every class `c` for a fixed document `d`.  
So the ranking only depends on the numerator `P(c) * P(d|c)`.

---

# Q4. Add-1 Smoothing
Given:
- Priors: `P(-)=3/5`, `P(+)=2/5`
- Vocabulary size `V = 20`
- Negative total token count = `14`

1) Denominator for negative likelihood with add-1:
- **14 + 20 = 34**

2) `P(predictable | -)` when count is 2:
- **(2 + 1) / 34 = 3/34**

3) `P(fun | -)` when count is 0:
- **(0 + 1) / 34 = 1/34**

---

# Q5. Programming Question — Tokenization

Implemented in `tokenization.py`.

Run:
```bash
python tokenization.py
```

## 1) Naïve space tokenization vs manual correction
- Naïve: `text.split()` (keeps punctuation attached, mishandles dashes, etc.)
- Manual: regex-based tokens that separate punctuation but keep contractions and hyphenated compounds.

The script prints both token lists and shows differences.

## 2) Compare with a tool
Tool used: **NLTK `wordpunct_tokenize`** (works offline without downloading models).  
The script prints tool tokens and differences vs manual.  
Typical differences happen because tool tokenizers split contractions or punctuation slightly differently (e.g., `"can't"` → `["can", "'", "t"]`).

## 3) Multiword Expressions (MWEs)
Examples used:
- `New York City`
- `Central Park`
- `machine learning`

They should often be treated as single tokens because they represent one concept / named entity and splitting them can break meaning (NER, phrase modeling, search).

## 4) Reflection (5–6 sentences)
Tokenization is hardest when punctuation, contractions, and hyphenated compounds appear together.  
English is easier than many morphologically rich languages, but it still has tricky cases like apostrophes (`can't`) and multiword named entities (`New York City`).  
Dashes and quotes also add ambiguity: sometimes they are punctuation, sometimes they glue words.  
MWEs are challenging because meaning is stored at the phrase level rather than word level.  
In general, punctuation and MWEs increase tokenization difficulty because they require context beyond simple splitting rules.

---

## How to run everything
```bash
pip install -r requirements.txt
python regex_tasks.py
python bpe.py --toy --merges 10
python bpe.py --paragraph --merges 30
python tokenization.py
```
