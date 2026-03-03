# PCL EDA Findings

**Dataset:** Don't Patronize Me! (DPM) — SemEval-2022 Task 4, Subtask 1
**Task:** Binary classification of Patronizing and Condescending Language (PCL vs No-PCL)
**Dataset size:** 10,469 paragraphs from English-language news, sourced from the NOW corpus
**Notebook:** `eda_pcl.ipynb` | **Figures:** `figures/`

---

## Dataset Overview

Each paragraph was retrieved by searching news articles for mentions of ten vulnerable-community
keywords (disabled, homeless, hopeless, immigrant, in-need, migrant, poor-families, refugee,
vulnerable, women). Two annotators independently rated each paragraph on a 0–4 scale; for
Task 1, labels 0–1 are collapsed to **No-PCL (0)** and labels 2–4 to **PCL (1)**.

---

## EDA 1 — Class Distribution & Label Granularity

### Visual Evidence
![Class Distribution](figures/fig1_class_distribution.png)

| Label | Description | Count | % |
|-------|-------------|------:|--:|
| 0 | No PCL | 9,476 | 90.5 % |
| 1 | PCL | 993 | 9.5 % |

Fine-grained breakdown (original 0–4 scale):

| Orig. label | Description | Count | % |
|-------------|-------------|------:|--:|
| 0 | Both annotators: No PCL | 8,529 | 81.5 % |
| 1 | Mixed: No PCL / Borderline | 947 | 9.0 % |
| 2 | Both annotators: Borderline PCL | 144 | 1.4 % |
| 3 | Mixed: Borderline / Strong PCL | 458 | 4.4 % |
| 4 | Both annotators: Strong PCL | 391 | 3.7 % |

### Analysis
The dataset is **severely imbalanced** at a 9.54:1 ratio (No-PCL to PCL). Crucially, the
fine-grained distribution reveals that the majority of PCL examples sit at labels 3 and 4
(strong / clear-cut PCL), with very few borderline cases at label 2. The binary collapse means
the PCL class is dominated by unambiguous examples, which is advantageous for learning a
clean decision boundary — but the sheer scarcity of the positive class remains the central
modelling challenge.

### Impact Statement
A naïve majority-class classifier would achieve ~90.5% accuracy while predicting No-PCL for
every sample, making accuracy a misleading metric. **F1-score on the positive class** must be
the primary evaluation metric. Class imbalance mandates one or more mitigation strategies:
weighted cross-entropy loss (weight ≈ 9.5 for the PCL class), oversampling PCL examples (e.g.
SMOTE on embeddings, or simply up-sampling), or threshold tuning at inference time. The scarcity
of label-2 examples (borderline, low-agreement) also suggests that uncertainty-aware methods may
be beneficial.

---

## EDA 2 — Text Length Analysis

### Visual Evidence
![Text Length](figures/fig2_text_length.png)

| Statistic | No-PCL (word count) | PCL (word count) |
|-----------|--------------------:|----------------:|
| Mean | 47.9 | 53.6 |
| Median | 42 | 47 |
| Max | 909 | 512 |
| Min | 0 | 6 |

| Percentile | Word count |
|-----------|----------:|
| 90th | 83 |
| 95th | 102 |
| 99th | 141 |

### Analysis
Both classes exhibit a **right-skewed distribution**, with most paragraphs between 20–100
words. PCL texts are marginally but consistently longer (mean 53.6 vs 47.9 words), suggesting
that patronising passages tend to be more elaborate — possibly because the author spends more
words contextualising or "justifying" the condescension. The long tail extends to 909 words for
No-PCL texts (likely full article quotes), while PCL texts top out at 512. The vast majority of
all texts (95%) fit within 102 words.

### Impact Statement
Setting `max_length = 128` tokens for a transformer model (e.g. BERT, RoBERTa) will capture
≥95% of the text without truncation, balancing coverage with computational cost. The few
extreme-length outliers (>200 words) should be inspected for data-collection artefacts; they
may warrant truncation or removal. The slight length difference between classes is unlikely to
be a useful standalone feature but may be included as auxiliary input.

---

## EDA 3 — Keyword / Vulnerable-Community Distribution

### Visual Evidence
![Keyword Distribution](figures/fig3_keyword_distribution.png)

| Keyword | No-PCL | PCL | Total | PCL Rate |
|---------|-------:|----:|------:|---------:|
| homeless | 899 | 178 | 1,077 | **16.5%** |
| poor-families | 759 | 150 | 909 | **16.5%** |
| in-need | 906 | 176 | 1,082 | **16.3%** |
| hopeless | 881 | 124 | 1,005 | 12.3% |
| refugee | 982 | 86 | 1,068 | 8.1% |
| disabled | 947 | 81 | 1,028 | 7.9% |
| vulnerable | 1,000 | 80 | 1,080 | 7.4% |
| women | 1,018 | 52 | 1,070 | 4.9% |
| migrant | 1,053 | 36 | 1,089 | 3.3% |
| immigrant | 1,031 | 30 | 1,061 | **2.8%** |

### Analysis
PCL rates vary dramatically across keywords — from **16.5%** (homeless, poor-families) down to
**2.8%** (immigrant). Communities associated with economic deprivation (homeless, poor-families,
in-need) attract substantially more patronising language than politically-framed groups
(immigrant, migrant). This likely reflects the nature of charity/aid journalism: articles about
economic poverty tend to adopt a "helpful saviour" tone, whereas immigration articles tend toward
political debate. Notably, the sample sizes are roughly balanced across keywords (~1,000–1,090
each), so the PCL rate differences are genuine and not artefacts of sampling.

### Impact Statement
The keyword a paragraph comes from is a **meaningful proxy for prior PCL probability** and
should be considered as an auxiliary feature (keyword embedding or one-hot indicator). Models
trained without this signal may underfit on low-rate keywords (immigrant, migrant) and
overfit on high-rate ones (homeless, poor-families). If the test set has a different keyword
distribution than training, performance may degrade — this motivates keyword-stratified
cross-validation during development. The data also hints that a keyword-conditioned threshold
could improve final F1.

---

## EDA 4 — N-gram Analysis (Lexical Patterns)

### Visual Evidence
![N-gram Analysis](figures/fig4_ngrams.png)

**Top-10 discriminative bigrams:**

| Rank | PCL Bigrams | No-PCL Bigrams |
|------|------------|----------------|
| 1 | poor families | poor families |
| 2 | **people need** | illegal immigrants |
| 3 | homeless people | per cent |
| 4 | **children poor** | united states |
| 5 | **help need** | homeless people |
| 6 | disabled people | disabled people |
| 7 | **vulnerable people** | sri lanka |
| 8 | men women | hong kong |
| 9 | women children | men women |
| 10 | **help people** | donald trump |

### Analysis
The n-gram analysis reveals a clear **rhetorical divergence** between the two classes.
PCL texts cluster around phrases of paternalistic helping: "people need", "help need",
"help people", "children poor" — language that centres the author's capacity to assist
rather than the agency of the affected community. No-PCL texts, by contrast, contain more
**geo-political and factual** phrases: "illegal immigrants", "per cent", "united states",
"donald trump", "sri lanka", "hong kong" — indicative of analytical or news-reporting
language. The shared bigrams (e.g. "homeless people", "disabled people") confirm that both
classes discuss the same communities, but through fundamentally different linguistic lenses.

### Impact Statement
Certain lexical patterns — particularly constructions involving "need", "help", and
community labels — are disproportionately associated with PCL. This suggests that:
1. **Stop-word lists should be conservative**: content words like "need" and "help" carry
   discriminative signal and must not be removed.
2. **Pre-trained word embeddings or language models** that encode semantic similarity will
   benefit from these distributional patterns already present at training time.
3. A TF-IDF + logistic regression baseline exploiting these n-grams should form a meaningful
   lower bound for more complex transformer models.

---

## EDA 5 — PCL Category Distribution & Co-occurrence

### Visual Evidence
![PCL Categories](figures/fig5_pcl_categories.png)

| Category | Count | % of PCL paragraphs |
|----------|------:|--------------------:|
| Unbalanced power relations | 716 | **72.1%** |
| Compassion | 469 | 47.2% |
| Authority voice | 230 | 23.2% |
| Presupposition | 224 | 22.6% |
| Metaphors | 197 | 19.8% |
| Shallow solution | 196 | 19.7% |
| Poorer the merrier | 40 | **4.0%** |

Average PCL categories per annotated paragraph: **2.09**

Strongest category co-occurrence: **Unbalanced power ↔ Shallow solution** (r = 0.22)

### Analysis
**Unbalanced power relations** is the dominant PCL category, present in nearly three-quarters
of all PCL paragraphs, suggesting that most patronising language operates through a power
dynamic where the author positions themselves as a benefactor. **Compassion** (47.2%) is the
second most common, often co-occurring with Unbalanced power — describing the vulnerable as
pitiable while the author stands ready to help. At the other extreme, **The poorer the merrier**
(4.0%) is rare, appearing in texts that romanticise poverty or hardship. The average of 2.09
categories per paragraph confirms that PCL is a **multi-faceted phenomenon**: most PCL texts
exhibit more than one rhetorical strategy simultaneously.

### Impact Statement
For **Subtask 1 (binary classification)**, the category distribution confirms that a model
must learn to recognise a wide variety of patronising strategies, not a single "trigger".
This favours contextual representations (transformers) over simple keyword matching.
For **Subtask 2 (multilabel classification)**, the extreme imbalance towards Unbalanced power
and away from Poorer the merrier demands **per-category class weighting** and evaluation via
per-category F1 rather than macro-averaged accuracy. The positive correlation between
Unbalanced power and Shallow solution also suggests that predicting one increases the
probability of the other — a multi-label model that captures label correlations (e.g. via
label-dependency chains) may outperform independent binary classifiers per category.

---

## EDA 6 — Word Cloud Visualisation

### Visual Evidence
![Word Clouds](figures/fig6_wordclouds.png)

*(TF-IDF weighted — larger words are more distinctive to that class, not merely more frequent)*

### Analysis
The TF-IDF weighting surfaces **characteristic vocabulary** rather than ubiquitous words.
The **PCL word cloud** is dominated by emotive community-focused language: *people*, *help*,
*poor*, *community*, *need*, *women*, *children*, *families*, *life* — words that enact a
compassionate, paternalistic register. The **No-PCL cloud** shows a markedly different
character: factual and institutional language (*government*, *percent*, *country*, *policy*,
*law*, *work*, *country*, *report*) alongside geographic referents, consistent with
straightforward news reporting. The contrast visually confirms the n-gram findings: PCL is
associated with "caring" relational language, while No-PCL is associated with analytical
reportage.

### Impact Statement
The vocabulary divergence is strong enough that even simple **bag-of-words features** should
show non-trivial performance. However, the many shared words (community labels, general news
vocabulary) mean that a model relying solely on individual word presence will conflate benign
reporting about vulnerable groups with genuinely patronising content. This motivates the use
of **contextual embeddings** (e.g. RoBERTa) that encode how words are used together, not just
which words appear.

---

## Summary: Key Modelling Implications

| Finding | Implication |
|---------|-------------|
| **9.54:1 class imbalance** | Use F1 (positive class) as primary metric; apply weighted loss or oversampling |
| **95% of texts ≤ 102 words** | Set `max_length = 128` for transformer tokenisers |
| **PCL rate varies 2.8%–16.5% by keyword** | Include keyword as auxiliary feature; use keyword-stratified CV |
| **PCL language centres "helping"** | Preserve content words (need, help) in any preprocessing |
| **Unbalanced power dominates PCL (72%)** | Binary model must learn diverse patronising strategies, not one trigger |
| **Avg 2.09 categories per PCL text** | Subtask 2 benefits from label-dependency modelling |
| **Clear vocabulary divergence (TF-IDF)** | TF-IDF + LR baseline is a meaningful lower bound; contextual models will improve further |
