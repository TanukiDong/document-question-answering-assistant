import json
import re
import time
from html import unescape
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


INPUT_CSV = "data/nq_raw.csv"
OUTPUT_JSON = "data/input_nq.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NQ-to-inputjson/1.0)"
}


def ensure_question_mark(q: str) -> str:
    q = str(q).strip()
    if not q.endswith("?"):
        q += "?"
    return q


def clean_golden_answer(ans: str) -> str:
    ans = str(ans).strip()

    # remove wrapping quotes like:
    # 'Nicholas Scott "Nick" Cannon'
    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in {"'", '"'}:
        ans = ans[1:-1].strip()

    return ans


def extract_title_and_oldid(wiki_url: str):
    parsed = urlparse(wiki_url)
    qs = parse_qs(parsed.query)

    title = qs.get("title", [None])[0]
    oldid = qs.get("oldid", [None])[0]

    if title is None or oldid is None:
        raise ValueError(f"Could not extract title/oldid from URL: {wiki_url}")

    return title, oldid


# def clean_wikipedia_text(text: str) -> str:
#     text = unescape(text)

#     # normalize whitespace
#     text = text.replace("\xa0", " ")
#     text = re.sub(r"\[[0-9]+\]", "", text)          # remove citation markers like [1]
#     text = re.sub(r"\s+", " ", text)                # collapse spaces
#     text = re.sub(r"\n\s*\n+", "\n\n", text)        # normalize blank lines
#     return text.strip()


def fetch_wikipedia_context_from_oldid(wiki_url: str, timeout: int = 20) -> str:
    """
    Fetch the rendered HTML for the exact old revision and extract paragraph text.
    This is usually easier to turn into clean plain text than raw wikitext.
    """
    title, oldid = extract_title_and_oldid(wiki_url)

    html_url = "https://en.wikipedia.org/w/index.php"
    params = {
        "title": title,
        "oldid": oldid,
    }

    response = requests.get(html_url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Main article content area
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        raise ValueError(f"Could not locate article content for URL: {wiki_url}")

    paragraphs = []
    for p in content.find_all("p", recursive=True):
        text = p.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)

    # Fallback: sometimes pages have very few <p> tags
    if not paragraphs:
        paragraphs = [
            tag.get_text(" ", strip=True)
            for tag in content.find_all(["p", "li"])
            if tag.get_text(" ", strip=True)
        ]

    document = "\n\n".join(paragraphs)
    # document = clean_wikipedia_text(document)

    if not document:
        raise ValueError(f"Extracted empty document for URL: {wiki_url}")

    return document

def convert_csv_to_input_json(input_csv: str, output_json: str):
    df = pd.read_csv(input_csv)

    records = []
    failed_rows = []

    for idx, row in tqdm(enumerate(df.itertuples()), total=len(df), desc="Processing"):
        question_id = f"q{idx + 1}"
        question = ensure_question_mark(row.question)
        golden_answer = clean_golden_answer(row.golden_answer)
        url = str(row.url).strip()

        try:
            document = fetch_wikipedia_context_from_oldid(url)

            records.append({
                "question_id": question_id,
                "question": question,
                "document": document,
                "golden_answer": [golden_answer]
            })

        except Exception as e:
            failed_rows.append({
                "row_index": idx,
                "question_id": question_id,
                "url": url,
                "error": str(e)
            })

        time.sleep(0.2)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records to {output_json}")


if __name__ == "__main__":
    convert_csv_to_input_json(INPUT_CSV, OUTPUT_JSON)