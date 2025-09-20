
# mood_of_india_streamlit.py
# Robust build: auto-installs snscrape (compatible versions), then uses Python API with subprocess fallback + diagnostics.

import os
import re
import sys
import json
import math
import html
import time
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Mood of India (Twitter/X)", page_icon="🇮🇳", layout="wide")

st.title("🇮🇳 Mood of India — Twitter/X Topic Sentiment")
st.caption("Fetch tweets on a topic and estimate support / oppose / neutral. This build auto-installs snscrape (compatible versions).")

with st.expander("Read me"):
    st.markdown("""
**What this does:**  
- Collects tweets via **snscrape** Python API (preferred). If import fails, the app will **auto-install** snscrape (tries unpinned, then `0.7.0.20230622`) and retry.  
- If Python API still fails, tries a short **subprocess** fallback (also requires snscrape).  
- Sentiment: VADER + small Hindi/Hinglish lexicon by default (lightweight).

**If it seems stuck:**  
- This version has a **60s hard timeout** for each collector and prints **diagnostics** below.
""")

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
WHITESPACE_RE = re.compile(r'\s+')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = URL_RE.sub('', s)
    s = MENTION_RE.sub('', s)
    s = HASHTAG_RE.sub(lambda m: m.group(0)[1:], s)
    s = s.replace('&amp;', '&')
    s = WHITESPACE_RE.sub(' ', s).strip()
    return s

# ---------------------------
# Ensure snscrape is present
# ---------------------------
def _pip_install(args, timeout=180):
    proc = subprocess.run([sys.executable, "-m", "pip"] + args,
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")

def ensure_snscrape(diag: dict) -> str:
    """
    Tries to import snscrape; if missing, tries:
      1) pip install --upgrade pip (once)
      2) pip install snscrape
      3) pip install snscrape==0.7.0.20230622
    Returns "" on success, or an error string.
    """
    try:
        import snscrape  # noqa: F401
        return ""
    except Exception:
        pass

    # 1) Upgrade pip (best-effort)
    rc, log = _pip_install(["install", "--upgrade", "pip"], timeout=120)
    diag["pip_upgrade_rc"] = rc
    if log.strip():
        diag["pip_upgrade_log"] = log[-2000:]  # last 2k chars

    # 2) Try unpinned snscrape
    rc, log = _pip_install(["install", "snscrape"], timeout=180)
    diag["pip_snscrape_unpinned_rc"] = rc
    if log.strip():
        diag["pip_snscrape_unpinned_log"] = log[-2000:]
    if rc == 0:
        try:
            import snscrape  # noqa: F401
            return ""
        except Exception as e:
            diag["snscrape_import_after_unpinned"] = str(e)

    # 3) Try pinned known-good
    rc, log = _pip_install(["install", "snscrape==0.7.0.20230622"], timeout=180)
    diag["pip_snscrape_pinned_rc"] = rc
    if log.strip():
        diag["pip_snscrape_pinned_log"] = log[-2000:]
    if rc == 0:
        try:
            import snscrape  # noqa: F401
            return ""
        except Exception as e:
            return f"import after pinned install failed: {e}"

    return "pip install snscrape failed (both unpinned and pinned). See Diagnostics for logs."

# ---------------------------
# SNSCRAPE COLLECTORS
# ---------------------------
def run_snscrape_python(query: str, lang: str, since: str, limit: int, india_only: bool=True, timeout_s: int=60) -> pd.DataFrame:
    """
    snscrape Python API path (preferred). Hard-stops after timeout_s seconds.
    """
    start = time.time()
    try:
        from snscrape.modules.twitter import TwitterSearchScraper
    except Exception as e:
        raise RuntimeError(f"snscrape import failed: {e}")

    q_parts = [query]
    if lang: q_parts.append(f'lang:{lang}')
    if since: q_parts.append(f'since:{since}')
    if india_only: q_parts.append('place_country:IN')
    q = " ".join(q_parts)

    rows = []
    try:
        scraper = TwitterSearchScraper(q)
        for i, tweet in enumerate(scraper.get_items()):
            if (i+1) > int(limit): break
            rows.append({
                'date': tweet.date,
                'username': getattr(tweet.user, 'username', None),
                'content': getattr(tweet, 'rawContent', getattr(tweet, 'content', None)),
                'likeCount': getattr(tweet, 'likeCount', None),
                'retweetCount': getattr(tweet, 'retweetCount', None),
                'replyCount': getattr(tweet, 'replyCount', None),
            })
            if time.time() - start > timeout_s:
                break
    except Exception as e:
        raise RuntimeError(f"snscrape Python API error: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['content'])
    return df

def run_snscrape_subprocess(query: str, lang: str, since: str, limit: int, india_only: bool=True, timeout_s: int=60) -> (pd.DataFrame, str):
    """
    Fallback: snscrape via subprocess with timeout.
    Returns dataframe + stderr for diagnostics.
    """
    q_parts = [query]
    if lang: q_parts.append(f'lang:{lang}')
    if since: q_parts.append(f'since:{since}')
    if india_only: q_parts.append('place_country:IN')
    q = " ".join(q_parts)

    cmd = [sys.executable, "-m", "snscrape", "--jsonl", "--max-results", str(int(limit)), "twitter-search", q]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return pd.DataFrame(), (proc.stderr or proc.stdout)
        lines = proc.stdout.strip().splitlines()
    except Exception as e:
        return pd.DataFrame(), f"subprocess error: {e}"

    rows = []
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        rows.append({
            'date': obj.get('date'),
            'username': (obj.get('user') or {}).get('username'),
            'content': obj.get('content'),
            'likeCount': obj.get('likeCount'),
            'retweetCount': obj.get('retweetCount'),
            'replyCount': obj.get('replyCount'),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['content'])
    return df, ""

# ---------------------------
# SENTIMENT (VADER default)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_vader():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

HINDI_LEXICON = {
    "vote chori": -0.9, "वोट चोरी": -0.9, "धांधली": -0.8, "scam": -0.6,
    "bharosa": 0.4, "भरोसा": 0.4, "badhiya": 0.5, "अच्छा": 0.5,
    "chor": -0.7, "झूठ": -0.7, "andhbhakt": -0.4,
    "कांग्रेस": 0.0, "भाजपा": 0.0,
}
EMOJI_POLARITY = {
    "😡": -0.8, "🤬": -0.9, "😭": -0.6, "😢": -0.5, "😞": -0.4, "😤": -0.4,
    "👍": 0.5, "🙏": 0.3, "🔥": 0.2, "🎉": 0.7, "😊": 0.5, "😀": 0.5, "😂": 0.2,
}
def lexicon_score(text: str) -> float:
    s = text.lower()
    score = 0.0
    for k, v in HINDI_LEXICON.items():
        if k in s:
            score += v
    for ch in text:
        if ch in EMOJI_POLARITY:
            score += EMOJI_POLARITY[ch]
    return max(-1.0, min(1.0, score))

def classify_sentiment(texts: List[str]) -> List[str]:
    sia = get_vader()
    out = []
    for t in texts:
        lt = clean_text(t)
        base = 0.0
        if sia is not None:
            base = sia.polarity_scores(lt)['compound']
        base += 0.4 * lexicon_score(t)
        if base > 0.15: out.append('support')
        elif base < -0.15: out.append('oppose')
        else: out.append('neutral')
    return out

def compute_weights(df: pd.DataFrame, cap_per_user: int, boost_engagement: bool=True) -> np.ndarray:
    if df.empty:
        return np.array([])
    counts = df['username'].value_counts().to_dict()
    def weight_row(row):
        base = 1.0 / max(1, counts.get(row['username'], 1) / cap_per_user)
        if boost_engagement:
            eng = (row.get('likeCount') or 0) + (row.get('retweetCount') or 0) + (row.get('replyCount') or 0)
            base *= math.log1p(eng + 1)
        return base
    return df.apply(weight_row, axis=1).values

# ---------------------------
# UI
# ---------------------------
with st.sidebar:
    st.header("Query")
    topic = st.text_input("Topic / keywords", value="vote chori")
    lang = st.selectbox("Language filter", ["", "en", "hi"], index=1, help="Leave blank for all languages.")
    days = st.slider("Lookback window (days)", 1, 30, value=2)
    limit = st.slider("Max tweets to fetch", 100, 10000, value=600, step=100)
    strict_india = st.checkbox("Strict India filter (place_country:IN)", value=True, help="ON = restrict to tweets with place metadata in India. OFF = global.")
    st.header("Analysis")
    cap_per_user = st.slider("Bot dampening cap (tweets/user)", 1, 50, 5)
    boost_eng = st.checkbox("Boost by engagement (likes/RT/replies)", value=False)
    st.header("Run")
    do_run = st.button("Fetch & Analyze", type="primary")

diag = {"collector": None, "q": None, "error": None, "counts": 0, "path": [], "py": sys.version.split()[0]}
df = pd.DataFrame()

if do_run:
    if not topic.strip():
        st.warning("Please enter a topic/keyword.")
        st.stop()
    since_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

    # Ensure snscrape is available (auto-install if needed)
    install_err = ensure_snscrape(diag)
    if install_err:
        diag["error"] = (diag.get("error","") + f"\n{install_err}").strip()

    # Try snscrape Python API
    diag["collector"] = "snscrape-python"
    diag["q"] = f'{topic} {"lang:"+lang if lang else ""} since:{since_date} {"place_country:IN" if strict_india else ""}'.strip()
    try:
        df = run_snscrape_python(topic, lang if lang else None, since_date, limit, india_only=strict_india, timeout_s=60)
        diag["counts"] = len(df)
        diag["path"].append("snscrape-python")
    except Exception as e:
        diag["error"] = (diag.get("error","") + f"\n{e}").strip()
        # Fallback: subprocess (also needs snscrape installed)
        try:
            diag["collector"] = "snscrape-subprocess"
            df, stderr = run_snscrape_subprocess(topic, lang if lang else None, since_date, limit, india_only=strict_india, timeout_s=60)
            diag["counts"] = len(df)
            diag["path"].append("snscrape-subprocess")
            if stderr:
                diag["error"] = (diag.get("error","") + "\n" + stderr).strip()
        except Exception as e2:
            diag["error"] = (diag.get("error","") + f"\nsubprocess fallback error: {e2}").strip()

    # Handle empty or failure
    if df.empty:
        st.error("No tweets fetched.")
        with st.expander("Diagnostics"):
            st.json(diag)
        st.stop()

    st.success(f"Fetched {len(df)} tweets via {diag['collector']}.")
    with st.expander("Diagnostics"):
        st.json(diag)

    df['clean'] = df['content'].astype(str).map(clean_text)
    df = df.drop_duplicates(subset=['clean']).dropna(subset=['clean'])

    labels = classify_sentiment(df['clean'].tolist())
    df['stance'] = labels
    w = compute_weights(df, cap_per_user=cap_per_user, boost_engagement=boost_eng)
    df['weight'] = w if len(w) == len(df) else 1.0

    grouped = df.groupby('stance')['weight'].sum()
    total_w = grouped.sum() if grouped.sum() else 1.0
    shares = (grouped / total_w).reindex(['support','oppose','neutral']).fillna(0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Support (weighted)", f"{shares.get('support',0)*100:.1f}%")
    col2.metric("Oppose (weighted)", f"{shares.get('oppose',0)*100:.1f}%")
    col3.metric("Neutral (weighted)", f"{shares.get('neutral',0)*100:.1f}%")
    col4.metric("Sample size", f"{len(df)} tweets")

    if 'date' in df.columns and not df['date'].isna().all():
        df['date_round'] = pd.to_datetime(df['date'], errors='coerce').dt.floor('H')
        trend = df.groupby(['date_round','stance'])['weight'].sum().reset_index()
        trend_pivot = trend.pivot(index='date_round', columns='stance', values='weight').fillna(0.0)
        st.subheader("Trend over time (weighted)")
        st.line_chart(trend_pivot)

    from collections import Counter
    def tokenize(s):
        return [w for w in re.findall(r"[#@\\w\\p{L}']+", s, flags=re.UNICODE) if len(w) > 2]
    try:
        toks_support = Counter([w.lower() for s in df[df['stance']=='support']['clean'] for w in tokenize(s)])
        toks_oppose  = Counter([w.lower() for s in df[df['stance']=='oppose']['clean']  for w in tokenize(s)])
        toks_neutral = Counter([w.lower() for s in df[df['stance']=='neutral']['clean'] for w in tokenize(s)])
    except re.error:
        simple_token = re.compile(r"[A-Za-z0-9_#@']+")
        toks_support = Counter([w.lower() for s in df[df['stance']=='support']['clean'] for w in simple_token.findall(s)])
        toks_oppose  = Counter([w.lower() for s in df[df['stance']=='oppose']['clean']  for w in simple_token.findall(s)])
        toks_neutral = Counter([w.lower() for s in df[df['stance']=='neutral']['clean'] for w in simple_token.findall(s)])

    topic_terms = set([w.lower() for w in re.findall(r"[A-Za-z\\p{L}']+", topic)])
    for c in (toks_support, toks_oppose, toks_neutral):
        for t in list(topic_terms):
            c.pop(t, None)

    def top_k(counter, k=20):
        return pd.DataFrame(counter.most_common(k), columns=['term','count']) if counter else pd.DataFrame(columns=['term','count'])

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Top terms — Support**")
        st.dataframe(top_k(toks_support, 20), use_container_width=True, hide_index=True)
    with colB:
        st.markdown("**Top terms — Oppose**")
        st.dataframe(top_k(toks_oppose, 20), use_container_width=True, hide_index=True)
    with colC:
        st.markdown("**Top terms — Neutral**")
        st.dataframe(top_k(toks_neutral, 20), use_container_width=True, hide_index=True)

    with st.expander("Show raw tweets"):
        st.dataframe(df[['date','username','content','stance','likeCount','retweetCount','replyCount']].sort_values('date', ascending=False), use_container_width=True)

    df_out = df[['date','username','content','stance','likeCount','retweetCount','replyCount','weight']].copy()
    st.download_button("Download results (CSV)", data=df_out.to_csv(index=False).encode('utf-8'),
                       file_name=f"tweets_{topic.replace(' ','_')}.csv", mime="text/csv")
