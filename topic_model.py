#!/usr/bin/env python3
"""
Temporal topic modeling over the 161 extracted Enemy Archives documents.

Pipeline:
1. Assign each document a year (extracted from title/opening lines;
   interpolated from neighboring document numbers for the ~6 undated ones,
   since the book's document numbering is itself chronological).
2. TF-IDF vectorize the document bodies (unigrams+bigrams, corpus-wide
   stopwords like "OUN"/"NKVD"/years removed since they don't discriminate
   between topics).
3. NMF topic model (offline, no model downloads needed - works within
   this sandbox's network restrictions, unlike embedding-based approaches
   that need to fetch pretrained weights).
4. Aggregate per-document topic weights by year to show topic volume
   over time, and export everything as JSON for the visualization.
"""

import json
import re
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sktext
from sklearn.decomposition import NMF

DOCS_DIR = "/home/claude/extracted_documents"
N_TOPICS = 10

TOPIC_LABELS = {
    0: "Combat Operations (German Front)",
    1: "Anti-Bolshevik Ideology & World Struggle",
    2: "Underground Security & Command",
    3: "UPA Military Command Structure",
    4: "Field Casualty & Raid Reports",
    5: "Ukrainian–Polish Conflict Operations",
    6: "Soviet Counterinsurgency (Agentura)",
    7: "National Political Program",
    8: "Collective Farms & Soviet Rule",
    9: "Individual Case Files & Hideouts",
}

YEAR_PAT = re.compile(r'(19[2-6]\d)')
YEAR_RANGE_PAT = re.compile(r'(19[2-6]\d)\s*[-\u2013]\s*(\d{2,4})')


def extract_year(title, body_start):
    """Prefer a year-range if present (e.g. "1944-55", "1949-50") - these are
    typically retrospective/summary reports compiled at or after the end of
    the period they describe, so use the range's END year rather than the
    first year regex would otherwise grab. Falls back to a plain single-year
    search (title first, then body opening) when no range is present."""
    for src in (title, body_start):
        m = YEAR_RANGE_PAT.search(src)
        if m:
            start_year = int(m.group(1))
            end_raw = m.group(2)
            if len(end_raw) == 4:
                end_year = int(end_raw)
            else:
                end_year = (start_year // 100) * 100 + int(end_raw)
                if end_year < start_year:
                    end_year += 100
            return end_year
    m2 = YEAR_PAT.search(title) or YEAR_PAT.search(body_start)
    return int(m2.group(1)) if m2 else None


def load_index():
    with open(f"{DOCS_DIR}/_index.json", encoding="utf-8") as f:
        idx = json.load(f)
    return sorted(idx, key=lambda d: d["number"])


def assign_years(idx):
    years = {}
    for d in idx:
        text = open(f"{DOCS_DIR}/{d['filename']}", encoding="utf-8").read()
        year = extract_year(d["title"], text[:800])
        if year is not None:
            years[d["number"]] = [year, False]

    nums = [d["number"] for d in idx]
    for n in nums:
        if n not in years:
            before = max([k for k in years if k < n], default=None)
            after = min([k for k in years if k > n], default=None)
            if before is not None and after is not None:
                yb, ya = years[before][0], years[after][0]
                est = round(yb + (ya - yb) * (n - before) / (after - before))
            elif before is not None:
                est = years[before][0]
            else:
                est = years[after][0]
            years[n] = [est, True]
    return years


def load_texts(idx):
    texts = []
    for d in idx:
        t = open(f"{DOCS_DIR}/{d['filename']}", encoding="utf-8").read()
        t = "\n".join(t.split("\n")[3:])  # drop our own "Document N: Title" header
        texts.append(t)
    return texts


def run_topic_model(texts):
    extra_stop = [
        "oun", "upa", "ukrainian", "ukrainians", "ukraine", "organization",
        "leadership", "soviet", "nkvd", "ngb", "mgb", "kgb", "nkgb", "document",
    ] + [str(y) for y in range(1929, 1956)]
    combined_stop = list(sktext.ENGLISH_STOP_WORDS.union(extra_stop))

    vec = TfidfVectorizer(max_df=0.6, min_df=3, stop_words=combined_stop, ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=500)
    W = nmf.fit_transform(X)   # doc x topic weights
    H = nmf.components_        # topic x term weights

    feature_names = vec.get_feature_names_out()
    topic_top_words = []
    for topic in H:
        top_idx = topic.argsort()[-12:][::-1]
        topic_top_words.append([feature_names[j] for j in top_idx])

    # normalize each doc's topic weights to sum to 1 (share of document)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W / row_sums

    return W_norm, topic_top_words


def main():
    idx = load_index()
    years = assign_years(idx)
    texts = load_texts(idx)
    W_norm, topic_top_words = run_topic_model(texts)

    documents = []
    for i, d in enumerate(idx):
        year, estimated = years[d["number"]]
        weights = W_norm[i].tolist()
        dominant = int(max(range(N_TOPICS), key=lambda k: weights[k]))
        documents.append({
            "number": d["number"],
            "title": d["title"],
            "year": year,
            "year_estimated": estimated,
            "word_count": d["word_count"],
            "dominant_topic": dominant,
            "topic_weights": weights,
        })

    year_range = list(range(min(years[n][0] for n in years), max(years[n][0] for n in years) + 1))
    year_topic_matrix = {y: [0.0] * N_TOPICS for y in year_range}
    year_doc_count = {y: 0 for y in year_range}
    for doc in documents:
        y = doc["year"]
        year_doc_count[y] += 1
        for k in range(N_TOPICS):
            year_topic_matrix[y][k] += doc["topic_weights"][k]

    topics = [{
        "id": k,
        "label": TOPIC_LABELS.get(k, f"Topic {k}"),
        "top_words": topic_top_words[k],
    } for k in range(N_TOPICS)]

    output = {
        "topics": topics,
        "years": year_range,
        "year_doc_count": year_doc_count,
        "year_topic_matrix": year_topic_matrix,
        "documents": documents,
    }

    with open("/home/claude/analysis/topic_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    print(f"{len(documents)} documents, {len(topics)} topics, years {year_range[0]}-{year_range[-1]}")
    print(f"Estimated (interpolated) years: {sum(1 for d in documents if d['year_estimated'])}")


if __name__ == "__main__":
    main()
