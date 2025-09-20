
# mood_of_india_streamlit.py (with CSV fallback)
# Prefers snscrape (if present) -> Nitter fallback -> CSV URLs fallback

import sys, os, re, html as htmllib, json, math, time, urllib.parse, urllib.request, ssl, random
from datetime import datetime, timedelta
from typing import List, Tuple
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mood of India (Twitter/X)", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Mood of India — Twitter/X Topic Sentiment")

with st.expander("Read me"):
    st.markdown("""
Paths tried in order:
1. **snscrape** Python API (if installed on host)
2. **Nitter** HTML fallback (no installs) with multiple mirrors + proxy
3. **CSV URLs** fallback — paste links like:
   - `https://raw.githubusercontent.com/OWNER/REPO/BRANCH/data/latest_vote_chori.csv`
    """)

def have_snscrape() -> bool:
    try:
        import snscrape  # noqa: F401
        return True
    except Exception:
        return False

def run_snscrape_python(query: str, lang: str, since: str, limit: int) -> pd.DataFrame:
    from snscrape.modules.twitter import TwitterSearchScraper
    q_parts = [query]
    if lang: q_parts.append(f'lang:{lang}')
    if since: q_parts.append(f'since:{since}')
    q = " ".join(q_parts)
    rows=[]
    for i, tw in enumerate(TwitterSearchScraper(q).get_items()):
        rows.append({
            "date": getattr(tw, "date", None),
            "username": getattr(getattr(tw, "user", None), "username", None),
            "content": getattr(tw, "rawContent", getattr(tw, "content", "")),
            "likeCount": getattr(tw, "likeCount", 0),
            "retweetCount": getattr(tw, "retweetCount", 0),
            "replyCount": getattr(tw, "replyCount", 0),
        })
        if (i+1)>=int(limit): break
    df=pd.DataFrame(rows)
    if not df.empty:
        df["date"]=pd.to_datetime(df["date"], errors="coerce")
        df=df.dropna(subset=["content"])
    return df

# Nitter fallback (compact)
BASE_MIRRORS = [
    "https://nitter.net","https://nitter.poast.org","https://nitter.lacontrevoie.fr",
    "https://nitter.privacydev.net","https://nitter.sethforprivacy.com","https://nitter.altgr.xyz",
    "https://nitter.cz","https://nitter.esmailelbob.xyz","https://farside.link/nitter"
]
def fetch_url(url: str, timeout: int=20) -> Tuple[str, int]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="ignore"), resp.getcode()
TAG_RE = re.compile(r"<[^>]+>", re.DOTALL); WS_RE = re.compile(r"\s+")
def strip_tags(x: str) -> str: return WS_RE.sub(" ", TAG_RE.sub(" ", x)).strip()
def parse_nitter_search(html_text: str) -> list:
    items=[]; blocks = re.split(r'<div class="timeline-item[^"]*">', html_text)[1:]
    for block in blocks:
        muser = re.search(r'href="/([^"/]+)/status/\d+"', block) or re.search(r'href="/([^"/]+)"', block)
        username = muser.group(1) if muser else None
        mcontent = re.search(r'<div class="tweet-content[^"]*">(.+?)</div>', block, flags=re.DOTALL)
        content = strip_tags(mcontent.group(1)) if mcontent else ""
        mdate = re.search(r'datetime="([^"]+)"', block); date = mdate.group(1) if mdate else None
        def get_count(icon):
            m = re.search(fr'{icon}[^>]*></span>\s*<span[^>]*>([\d,\.]+)</span>', block)
            if not m: return 0
            s=m.group(1).replace(",","")
            try: return int(float(s))
            except: return 0
        likes=get_count("icon-heart"); rts=get_count("icon-retweet"); reps=get_count("icon-comment")
        if content:
            items.append({"date":date, "username":username, "content":htmllib.unescape(content),
                          "likeCount":likes, "retweetCount":rts, "replyCount":reps})
    return items
def run_nitter_search(query: str, lang: str, since: str, limit: int, diag: dict, custom: str=None) -> pd.DataFrame:
    q_parts=[query]; 
    if lang: q_parts.append(f"lang:{lang}")
    if since: q_parts.append(f"since:{since}")
    q_enc=urllib.parse.quote_plus(" ".join(q_parts))
    mirrors = ([custom.rstrip("/")] if custom else []) + BASE_MIRRORS
    rows=[]; errors=[]
    random.shuffle(mirrors)
    for base in mirrors:
        for mode in ["direct","proxy"]:
            try:
                if mode=="direct":
                    url=f"{base}/search?f=tweets&q={q_enc}"
                else:
                    inner = base.replace("https://","http://")
                    url=f"https://r.jina.ai/http://{inner.split('://')[1]}/search?f=tweets&q={q_enc}"
                html_text,_ = fetch_url(url, timeout=25)
                rows.extend(parse_nitter_search(html_text))
                if len(rows)>=limit: raise StopIteration
            except StopIteration:
                break
            except Exception as e:
                errors.append(f"{mode} {base}: {e}")
        if len(rows)>=limit: break
    if errors: diag["nitter_errors"]=" | ".join(errors)[-2000:]
    df=pd.DataFrame(rows[:limit])
    if not df.empty:
        df["date"]=pd.to_datetime(df["date"], errors="coerce"); df=df.dropna(subset=["content"])
    return df

# Sentiment (lightweight)
HINDI_LEXICON={"vote chori":-0.9,"वोट चोरी":-0.9,"धांधली":-0.8,"scam":-0.6,"bharosa":0.4,"भरोसा":0.4,"badhiya":0.5,"अच्छा":0.5,"chor":-0.7,"झूठ":-0.7,"andhbhakt":-0.4}
EMOJI_POLARITY={"😡":-0.8,"🤬":-0.9,"😭":-0.6,"😢":-0.5,"😞":-0.4,"😤":-0.4,"👍":0.5,"🙏":0.3,"🔥":0.2,"🎉":0.7,"😊":0.5,"😀":0.5,"😂":0.2}
def clean_text(s: str) -> str:
    s = htmllib.unescape(s or "")
    s = re.sub(r'https?://\S+|www\.\S+','',s); s = re.sub(r'@\w+','',s); s = re.sub(r'#(\w+)',r'\1',s)
    s = re.sub(r'\s+',' ',s).strip(); return s
def lexicon_score(text: str) -> float:
    s=(text or "").lower(); score=0.0
    for k,v in HINDI_LEXICON.items():
        if k in s: score+=v
    for ch in text:
        if ch in EMOJI_POLARITY: score+=EMOJI_POLARITY[ch]
    return max(-1.0,min(1.0,score))
def classify_sentiment(texts: list) -> list:
    out=[]; 
    for t in texts:
        base=0.4*lexicon_score(t)
        out.append("support" if base>0.1 else ("oppose" if base<-0.1 else "neutral"))
    return out
def compute_weights(df: pd.DataFrame, cap_per_user: int, boost_engagement: bool=True) -> np.ndarray:
    if df.empty: return np.array([])
    counts=df["username"].value_counts().to_dict()
    def wrow(row):
        base=1.0/max(1, counts.get(row.get("username"),1)/cap_per_user)
        if boost_engagement:
            eng=(row.get("likeCount") or 0)+(row.get("retweetCount") or 0)+(row.get("replyCount") or 0)
            base*= (1.0+(eng**0.3))
        return base
    return df.apply(wrow, axis=1).values

# UI
with st.sidebar:
    st.header("Query")
    topic = st.text_input("Topic / keywords", value="vote chori")
    lang = st.selectbox("Language filter", ["", "en", "hi"], index=1)
    days = st.slider("Lookback window (days)", 1, 7, value=2)
    limit = st.slider("Max tweets to fetch", 100, 2000, value=600, step=100)
    custom_nitter = st.text_input("Custom Nitter base (optional)", value="")
    csv_urls = st.text_area("Fallback CSV URLs (comma-separated)", value="", help="e.g., https://raw.githubusercontent.com/OWNER/REPO/BRANCH/data/latest_vote_chori.csv")
    st.header("Analysis")
    cap_per_user = st.slider("Bot dampening cap (tweets/user)", 1, 50, 5)
    boost_eng = st.checkbox("Boost by engagement (likes/RT/replies)", value=False)
    do_run = st.button("Fetch & Analyze", type="primary")

diag={"collector":None,"q":None,"error":"", "counts":0,"path":[],"py":sys.version.split()[0]}
df=pd.DataFrame()

if do_run:
    since_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    diag["q"]=f'{topic} {"lang:"+lang if lang else ""} since:{since_date}'.strip()

    if have_snscrape():
        try:
            diag["collector"]="snscrape-python"
            df = run_snscrape_python(topic, lang or None, since_date, limit)
            diag["counts"]=len(df); diag["path"].append("snscrape-python")
        except Exception as e:
            diag["error"] += f"\n{e}"

    if df.empty:
        diag["collector"]="nitter-fallback"
        df = run_nitter_search(topic, lang or None, since_date, limit, diag, custom=custom_nitter.strip() or None)
        diag["counts"]=len(df); diag["path"].append("nitter-fallback")

    if df.empty and csv_urls.strip():
        diag["collector"]="csv-fallback"
        rows=[]
        for url in [u.strip() for u in csv_urls.split(",") if u.strip()]:
            try:
                dfu = pd.read_csv(url)
                rows.append(dfu)
            except Exception as e:
                diag["error"] += f"\nCSV fetch failed: {url} -> {e}"
        if rows:
            df = pd.concat(rows, ignore_index=True)
            diag["counts"]=len(df); diag["path"].append("csv-fallback")

    if df.empty:
        st.error("No posts fetched (snscrape/Nitter/CSV all failed).")
        with st.expander("Diagnostics"): st.json(diag)
        st.stop()

    st.success(f"Fetched {len(df)} posts via {diag['collector']}.")
    with st.expander("Diagnostics"): st.json(diag)

    df["clean"]=df["content"].astype(str).map(clean_text)
    df=df.drop_duplicates(subset=["clean"]).dropna(subset=["clean"])

    labels=classify_sentiment(df["clean"].tolist()); df["stance"]=labels
    w=compute_weights(df, cap_per_user=cap_per_user, boost_engagement=boost_eng); df["weight"]=w if len(w)==len(df) else 1.0

    grouped=df.groupby("stance")["weight"].sum(); total=grouped.sum() or 1.0
    shares=(grouped/total).reindex(["support","oppose","neutral"]).fillna(0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Support (weighted)", f"{shares.get('support',0)*100:.1f}%")
    c2.metric("Oppose (weighted)", f"{shares.get('oppose',0)*100:.1f}%")
    c3.metric("Neutral (weighted)", f"{shares.get('neutral',0)*100:.1f}%")
    c4.metric("Sample size", f"{len(df)}")

    if "date" in df.columns and not df["date"].isna().all():
        df["date_round"]=pd.to_datetime(df["date"], errors="coerce").dt.floor("H")
        trend=df.groupby(["date_round","stance"])["weight"].sum().reset_index()
        pivot=trend.pivot(index="date_round", columns="stance", values="weight").fillna(0.0)
        st.subheader("Trend over time (weighted)")
        st.line_chart(pivot)

    from collections import Counter
    token_re = re.compile(r"[#@\w\p{L}']+", re.UNICODE)
    def toks(s):
        try: return [w for w in token_re.findall(s) if len(w)>2]
        except re.error: return re.findall(r"[A-Za-z0-9_#@']+", s)

    toks_support=Counter([w.lower() for s in df[df["stance"]=="support"]["clean"] for w in toks(s)])
    toks_oppose =Counter([w.lower() for s in df[df["stance"]=="oppose"]["clean"]  for w in toks(s)])
    toks_neutral=Counter([w.lower() for s in df[df["stance"]=="neutral"]["clean"] for w in toks(s)])

    colA,colB,colC=st.columns(3)
    def top_k(c,k=20): return pd.DataFrame(c.most_common(k), columns=["term","count"]) if c else pd.DataFrame(columns=["term","count"])
    with colA:
        st.markdown("**Top terms — Support**"); st.dataframe(top_k(toks_support), use_container_width=True, hide_index=True)
    with colB:
        st.markdown("**Top terms — Oppose**"); st.dataframe(top_k(toks_oppose), use_container_width=True, hide_index=True)
    with colC:
        st.markdown("**Top terms — Neutral**"); st.dataframe(top_k(toks_neutral), use_container_width=True, hide_index=True)

    with st.expander("Show raw posts"):
        st.dataframe(df[["date","username","content","stance","likeCount","retweetCount","replyCount"]].sort_values("date", ascending=False), use_container_width=True)
