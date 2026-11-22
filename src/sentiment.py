"""
Sentiment: FinBERT probabilities for news articles.

ADAPTER PATTERN: this is the only module that knows which model scores text. Everything else
consumes five numbers per article:
  p_pos, p_neg, p_neu   FinBERT's probabilities (sum to 1)
  s = p_pos - p_neg     signed tone, -1..+1 (|s| <= w always)
  w = 1 - p_neu         how much the article is "about" good/bad news at all, used as its weight

Input = headline + summary + full text (HTML already stripped by news.py), truncated to FinBERT's
512 tokens. The text is scored in memory and never stored.

The model revision is pinned so the backtest and the live bot use the identical model.
Scores can be cached by article id: the backtest scores ~60k articles once, and the live bot
keeps a small rolling cache in the repo so each morning only scores new articles.
"""
import datetime as dt
import json
import os

MODEL_ID = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MAX_TOKENS = 512

_model = None  # (tokenizer, model, label order), loaded on first use


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print("Loading FinBERT AI Model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model.eval()
        torch.manual_seed(0)
        # Use every core: some environments default torch to a single thread (10x slower here).
        torch.set_num_threads(os.cpu_count() or 1)
        labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
        _model = (tokenizer, model, labels)
    return _model


def model_input(article: dict) -> str:
    parts = [article.get("headline", "").strip().rstrip("."), article.get("summary", "").strip(),
             article.get("text", "").strip()]
    head, rest = parts[0], " ".join(p for p in parts[1:] if p)
    return f"{head}. {rest}" if rest else head


def score_texts(texts: list, batch_size: int = 32) -> list:
    """[{p_pos, p_neg, p_neu}, ...] for each text, in order."""
    import torch
    tokenizer, model, labels = _load()
    # Batch texts of similar length together: each batch is padded to its longest text.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                          return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(model(**batch).logits, dim=-1).tolist()
        for i, p in zip(idx, probs):
            d = dict(zip(labels, p))
            out[i] = {"p_pos": d["positive"], "p_neg": d["negative"], "p_neu": d["neutral"]}
    return out


def score_articles(articles: list, cache: dict | None = None, batch_size: int = 32) -> list:
    """Adds p_pos, p_neg, p_neu, s, w to each article (in place) and returns the list."""
    cache = {} if cache is None else cache
    todo = [a for a in articles if a["id"] not in cache]
    unique = list({a["id"]: a for a in todo}.values())  # one article can appear once per tagged stock
    for a, probs in zip(unique, score_texts([model_input(a) for a in unique], batch_size)):
        created = a.get("created_at")
        cache[a["id"]] = {**probs, "created_at": created.isoformat() if hasattr(created, "isoformat") else created}
    for a in articles:
        p = cache[a["id"]]
        a.update(p_pos=p["p_pos"], p_neg=p["p_neg"], p_neu=p["p_neu"],
                 s=p["p_pos"] - p["p_neg"], w=1 - p["p_neu"])
    return articles


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    if raw.get("model_revision") != MODEL_REVISION:
        return {}  # scores from another model version must not be mixed in
    return {int(k): v for k, v in raw["scores"].items()}


def save_cache(path: str, cache: dict, keep_since: dt.datetime) -> None:
    """Write the cache, keeping only articles published since keep_since (the baseline window)."""
    keep = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in cache.items()
            if v.get("created_at") and dt.datetime.fromisoformat(v["created_at"]) >= keep_since}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # One article per line, in id order: the file is committed daily, so a day's git diff is just the new and
    # pruned lines instead of one rewritten 1 MB line.
    lines = [f"{json.dumps(str(k))}:{json.dumps(keep[k], separators=(',', ':'))}" for k in sorted(keep, key=int)]
    head = json.dumps({"model_id": MODEL_ID, "model_revision": MODEL_REVISION})[:-1]
    with open(path, "w", newline="\n") as f:
        f.write(head + ',"scores":{\n' + ",\n".join(lines) + "\n}}\n")
