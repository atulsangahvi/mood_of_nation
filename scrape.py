
import os, sys, argparse
from datetime import datetime, timedelta
import pandas as pd

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=str, default="vote chori, stock market")
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    since_date = (datetime.utcnow() - timedelta(days=args.days)).date().isoformat()
    topics=[t.strip() for t in args.topics.split(",") if t.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    for topic in topics:
        df = run_snscrape_python(topic, args.lang or None, since_date, args.limit)
        safe = "_".join(topic.lower().split())
        stamp = datetime.utcnow().strftime("%Y%m%d")
        path_daily = os.path.join(args.outdir, f"{safe}_{stamp}.csv")
        path_latest = os.path.join(args.outdir, f"latest_{safe}.csv")
        df.to_csv(path_daily, index=False, encoding="utf-8")
        df.to_csv(path_latest, index=False, encoding="utf-8")
        print(f"Wrote {len(df)} rows for '{topic}' -> {path_daily} and {path_latest}")

if __name__ == "__main__":
    import pandas as pd  # ensure pandas present
    main()
