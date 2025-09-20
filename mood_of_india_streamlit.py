
# mood_of_india_streamlit.py
# Streamlit app: Query X/Twitter for a topic (e.g., "vote chori") and estimate support / oppose / neutral
# Data collectors: TWINT (if available) -> fallback to snscrape
# Sentiment: Transformers (cardiffnlp/twitter-xlm-roberta-base-sentiment) -> fallback to VADER + basic lexicon for Hindi/Hinglish
#
# How to run:
#   1) pip install -r requirements.txt
#   2) streamlit run mood_of_india_streamlit.py
#
# Cloud-safe defaults:
# - Transformer toggle OFF by default (uses VADER unless you enable it)
# - Prefer TWINT is OFF by default (uses snscrape)
# - limit=1000, days=2

import os
import re
import sys
import json
import time
import math
import html
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Mood of India (Twitter/X)", page_icon="🇮🇳", layout="wide")

st.title("🇮🇳 Mood of India — Twitter/X Topic Sentiment")
st.caption("Type a topic like **vote chori**, **stock market**, **inflation**, etc. The app will fetch recent tweets and estimate support/oppose/neutral.")

with st.expander("Read me"):
    st.markdown("""
**What this does:**  
- Collects tweets for your query via **TWINT** (no API key). If that fails or is disabled, uses **snscrape**.  
- Runs sentiment using a multilingual **transformer** (if enabled), else **VADER** + a small **Hindi/Hinglish** lexicon.  
- Shows a mood meter, trend, top keywords, and lets you download a CSV.

**Caveats:**  
- Twitter/X is not a representative sample of India. Expect bias and noise.  
- Bots and coordinated campaigns exist—use the "Bot dampening" slider to down-weight high-frequency posters.  
- Language and sarcasm are tricky; scores are probabilistic, not gospel.
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

def have_twint() -> bool:
    try:
        import twint  # noqa: F401
        return True
    except Exception:
        return False

def run_twint_search(query: str, lang: str, since: str, limit: int, near: str = None) -> pd.DataFrame:
    try:
        import twint
    except Exception as e:
        raise RuntimeError(f"TWINT not available: {e}")

    c = twint.Config()
    c.Search = query
    if lang:
        c.Lang = lang
    if since:
        c.Since = since
    if near:
        c.Near = near
    c.Limit = int(limit)
    c.Hide_output = True
    tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_json.close()
    c.Store_json = True
    c.Output = tmp_json.name
    try:
        twint.run.Search(c)
    except Exception as e:
        raise RuntimeError(f"TWINT error: {e}")
    rows = []
    with open(tmp_json.name, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append({
                'date': obj.get('date') or obj.get('created_at') or obj.get('created'),
                'username': obj.get('username') or obj.get('user',{}).get('username'),
                'content': obj.get('tweet') or obj.get('content') or obj.get('full_text'),
                'likeCount': obj.get('likes_count') or obj.get('likes') or obj.get('favorite_count'),
                'retweetCount': obj.get('retweets_count') or obj.get('retweets') or obj.get('retweet_count'),
                'replyCount': obj.get('replies_count') or obj.get('replies') or obj.get('reply_count'),
            })
    os.unlink(tmp_json.name)
    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['content'])
    return df

def run_snscrape_search(query: str, lang: str, since: str, limit: int, india_only: bool=True) -> pd.DataFrame:
    q_parts = [query]
    if lang:
        q_parts.append(f'lang:{lang}')
    if since:
        q_parts.append(f'since:{since}')
    if india_only:
        q_parts.append('place_country:IN')
    q = " ".join(q_parts)

    cmd = [
        sys.executable, "-m", "snscrape", "--jsonl",
        "--max-results", str(int(limit)),
        "twitter-search", q
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = proc.stdout.strip().splitlines()
    except Exception as e:
        raise RuntimeError(f"snscrape error: {e}")

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
    return df

def collect_tweets(query: str, lang: str, days: int, limit: int, geofocus: str, prefer_twint: bool=True) -> (pd.DataFrame, str):
    since_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    if prefer_twint and have_twint():
        try:
            df = run_twint_search(query, lang, since_date, limit, near=geofocus or None)
            if not df.empty:
                return df, "twint"
        except Exception as e:
            st.info(f"TWINT failed: {e}. Falling back to snscrape.")
    try:
        df = run_snscrape_search(query, lang, since_date, limit, india_only=True)
        if not df.empty:
            return df, "snscrape"
    except Exception as e:
        st.error(f"snscrape failed: {e}")
    return pd.DataFrame(), "none"

@st.cache_resource(show_spinner=False)
def get_transformer_pipeline():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        labels = ["negative","neutral","positive"]
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
        return pipe, labels
    except Exception as e:
        return None, None

@st.cache_resource(show_spinner=False)
def get_vader():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        return sia
    except Exception:
        return None

HINDI_LEXICON = {
    "vote chori": -0.9,
    "वोट चोरी": -0.9,
    "धांधली": -0.8,
    "scam": -0.6,
    "bharosa": 0.4,
    "भरोसा": 0.4,
    "badhiya": 0.5,
    "अच्छा": 0.5,
    "chor": -0.7,
    "झूठ": -0.7,
    "andhbhakt": -0.4,
    "कांग्रेस": 0.0,
    "भाजपा": 0.0,
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
    use_transformer = st.session_state.get('use_transformer', False)
    results = []
    pipe, labels = (None, None)
    if use_transformer:
        pipe, labels = get_transformer_pipeline()
        if pipe is None:
            st.info("Transformer model unavailable; falling back to VADER + lexicon.")
            use_transformer = False

    if use_transformer and pipe is not None:
        try:
            out = pipe(list(texts), truncation=True)
            for scores in out:
                top = max(scores, key=lambda d: d['score'])
                lab = top['label'].lower()
                if 'neg' in lab:
                    results.append('oppose')
                elif 'pos' in lab:
                    results.append('support')
                else:
                    results.append('neutral')
            return results
        except Exception:
            st.info("Transformer inference failed; switching to VADER + lexicon.")

    sia = get_vader()
    for t in texts:
        lt = clean_text(t)
        base = 0.0
        if sia is not None:
            vs = sia.polarity_scores(lt)['compound']
            base = vs
        base += 0.4 * lexicon_score(t)
        if base > 0.15:
            results.append('support')
        elif base < -0.15:
            results.append('oppose')
        else:
            results.append('neutral')
    return results

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
    w = df.apply(weight_row, axis=1).values
    return w

with st.sidebar:
    st.header("Query")
    topic = st.text_input("Topic / keywords", value="vote chori")
    lang = st.selectbox("Language filter", ["", "en", "hi"], index=2, help="Leave blank for all languages.")
    days = st.slider("Lookback window (days)", 1, 30, value=2)
    limit = st.slider("Max tweets to fetch", 100, 10000, value=1000, step=100)
    geofocus = st.text_input("Geofocus (TWINT 'near=')", value="Delhi", help="TWINT only; snscrape uses place_country:IN filter.")
    prefer_twint = st.checkbox("Prefer TWINT (fallback to snscrape)", value=False)
    st.header("Analysis")
    cap_per_user = st.slider("Bot dampening cap (tweets/user)", 1, 50, 5, help="Down-weights users who post more than this per window.")
    boost_eng = st.checkbox("Boost by engagement (likes/RT/replies)", value=True)
    use_transformer = st.checkbox("Use transformer model (heavier; faster OFF)", value=False)
    st.session_state['use_transformer'] = use_transformer
    st.header("Run")
    do_run = st.button("Fetch & Analyze", type="primary")

if do_run:
    if not topic.strip():
        st.warning("Please enter a topic/keyword.")
        st.stop()

    with st.status("Collecting tweets...", expanded=False) as status:
        df, source = collect_tweets(topic, lang if lang else None, days, limit, geofocus if geofocus else None, prefer_twint=prefer_twint)
        if df.empty:
            st.error("No tweets fetched. Try reducing filters, increasing lookback, or ensuring snscrape/twint are installed.")
            st.stop()
        status.update(label=f"Collected {len(df)} tweets via {source}. Cleaning…", state="running")
        df['clean'] = df['content'].astype(str).map(clean_text)
        df = df.drop_duplicates(subset=['clean'])
        df = df.dropna(subset=['clean'])
        status.update(label=f"{len(df)} unique tweets after cleaning. Classifying sentiment…", state="running")

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
        df['date_round'] = df['date'].dt.floor('H')
        trend = df.groupby(['date_round','stance'])['weight'].sum().reset_index()
        trend_pivot = trend.pivot(index='date_round', columns='stance', values='weight').fillna(0.0)
        trend_pivot = trend_pivot.reindex(columns=['support','oppose','neutral']).fillna(0.0)
        st.subheader("Trend over time (weighted)")
        st.line_chart(trend_pivot)
    else:
        st.info("No timestamps available to plot trend.")

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
    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download results (CSV)", data=csv_bytes, file_name=f"tweets_{topic.replace(' ','_')}.csv", mime="text/csv")
