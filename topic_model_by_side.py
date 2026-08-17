#!/usr/bin/env python3
"""
Run separate NMF topic models for the OUN/UPA-authored documents and the
Soviet-authored documents, and export data for a side-by-side temporal
visualization. Reuses the year-extraction/interpolation logic and the
same TF-IDF + NMF machinery as topic_model.py, but factors each side
independently so that e.g. "bandit"/"agentura" vocabulary doesn't just
absorb an entire topic and drown out finer Soviet-side distinctions,
and OUN ideological vocabulary doesn't get diluted by the larger corpus.
"""
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sktext
from sklearn.decomposition import NMF

DOCS_DIR = "/home/claude/extracted_documents"
YEAR_PAT = re.compile(r'(19[2-6]\d)')
YEAR_RANGE_PAT = re.compile(r'(19[2-6]\d)\s*[-\u2013]\s*(\d{2,4})')


def extract_year(title, body_start):
    """See topic_model.py for rationale: a year-range like "1944-55" in the
    title/opening lines usually marks a retrospective summary report, so we
    use the range's END year rather than the first year a plain regex would
    grab."""
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

N_TOPICS_OUN = 8
N_TOPICS_SOVIET = 6   # smaller corpus (37 docs) -> fewer topics to stay stable

EXTRA_STOP = [
    "oun", "upa", "ukrainian", "ukrainians", "ukraine", "organization",
    "leadership", "soviet", "nkvd", "ngb", "mgb", "kgb", "nkgb", "document",
] + [str(y) for y in range(1929, 1956)]


def load_index_and_sides():
    idx = json.load(open(f"{DOCS_DIR}/_index.json", encoding="utf-8"))
    sides = json.load(open("/home/claude/analysis/side_classification.json", encoding="utf-8"))
    side_map = {s["number"]: s["side"] for s in sides}
    idx = sorted(idx, key=lambda d: d["number"])
    for d in idx:
        d["side"] = side_map[d["number"]]
    return idx


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


def load_body(d):
    t = open(f"{DOCS_DIR}/{d['filename']}", encoding="utf-8").read()
    return "\n".join(t.split("\n")[3:])


def run_nmf(docs, n_topics):
    texts = [load_body(d) for d in docs]
    combined_stop = list(sktext.ENGLISH_STOP_WORDS.union(EXTRA_STOP))
    vec = TfidfVectorizer(max_df=0.7, min_df=2, stop_words=combined_stop, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vec.get_feature_names_out()
    topic_top_words = []
    for topic in H:
        top_idx = topic.argsort()[-12:][::-1]
        topic_top_words.append([feature_names[j] for j in top_idx])
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W / row_sums
    return W_norm, topic_top_words


def build_side_payload(docs, years, n_topics, labels):
    W_norm, topic_top_words = run_nmf(docs, n_topics)
    documents = []
    for i, d in enumerate(docs):
        year, estimated = years[d["number"]]
        weights = W_norm[i].tolist()
        dominant = int(max(range(n_topics), key=lambda k: weights[k]))
        documents.append({
            "number": d["number"], "title": d["title"], "year": year,
            "year_estimated": estimated, "word_count": d["word_count"],
            "dominant_topic": dominant, "topic_weights": weights,
        })
    year_range = list(range(min(y[0] for y in years.values()), max(y[0] for y in years.values()) + 1))
    ytm = {y: [0.0] * n_topics for y in year_range}
    ydc = {y: 0 for y in year_range}
    for doc in documents:
        y = doc["year"]
        ydc[y] += 1
        for k in range(n_topics):
            ytm[y][k] += doc["topic_weights"][k]
    topics = [{"id": k, "label": labels.get(k, f"Topic {k}"), "top_words": topic_top_words[k]}
              for k in range(n_topics)]
    return {
        "topics": topics, "years": year_range, "year_doc_count": ydc,
        "year_topic_matrix": ytm, "documents": documents,
    }


def main():
    idx = load_index_and_sides()
    years = assign_years(idx)

    oun_docs = [d for d in idx if d["side"] == "oun"]
    soviet_docs = [d for d in idx if d["side"] == "soviet"]
    other_docs = [d for d in idx if d["side"] == "other_german"]

    OUN_LABELS = {
        0: "Anti-Bolshevik World Struggle",
        1: "Combat Engagements",
        2: "UPA Command Structure",
        3: "Underground Organizational Directives",
        4: "Soviet Raids on Villages",
        5: "Ukrainian–Polish Conflict",
        6: "National Political Program",
        7: "Anti-German Resistance",
    }
    SOVIET_LABELS = {
        0: "Underground-Leadership Tracking",
        1: "Chekist Military Operations",
        2: "Raid & Detention Reports",
        3: "Countering the OUN's SB",
        4: "NKVD/MGB Command Orders",
        5: "Hideout Raids & Case Files",
    }

    oun_payload = build_side_payload(oun_docs, years, N_TOPICS_OUN, OUN_LABELS)
    soviet_payload = build_side_payload(soviet_docs, years, N_TOPICS_SOVIET, SOVIET_LABELS)

    other_documents = []
    for d in other_docs:
        year, estimated = years[d["number"]]
        other_documents.append({
            "number": d["number"], "title": d["title"], "year": year,
            "year_estimated": estimated, "word_count": d["word_count"],
        })

    output = {
        "oun": oun_payload,
        "soviet": soviet_payload,
        "other_german": other_documents,
        "counts": {"oun": len(oun_docs), "soviet": len(soviet_docs), "other_german": len(other_docs)},
    }
    with open("/home/claude/analysis/topic_data_by_side.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    print(f"OUN: {len(oun_docs)} docs / {N_TOPICS_OUN} topics")
    print(f"Soviet: {len(soviet_docs)} docs / {N_TOPICS_SOVIET} topics")
    print(f"Other (German): {len(other_docs)} docs, excluded from topic modeling (too few)")


if __name__ == "__main__":
    main()
