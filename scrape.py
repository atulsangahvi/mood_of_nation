import sys, argparse, time, random, traceback
from datetime import datetime, timedelta
import pandas as pd

def log(msg): print(f"[scrape] {msg}", flush=True)

def try_iter_items(scraper, max_retries=6, base_sleep=2.0):
    """Iterate scraper.get_items() with backoff + jitter on failures."""
    attempts = 0
    while True:
        try:
            for item in scraper.get_items():
                yield item
            return
        except Exception as e:
            attempts += 1
            if attempts > max_retries:
                raise
            sleep = base_sleep * (2 ** (attempts - 1)) + random.uniform(0, 0.75)
            log(f"Transient scraping error: {e} | retry {attempts}/{max_retries} after {sleep:.1f}s")
            time.sleep(sleep)

def run_slice(query: str, limit: int) -> pd.DataFrame:
    from snscrape.modules.twitter import TwitterSearchScraper
    rows = []
    i = 0
    log(f"Query => {query}")
    scraper = TwitterSearchScraper(query)
    for tw in try_iter_items(scraper):
        rows.append({
            "date": getattr(tw, "date", None),
            "username": getattr(getattr(tw, "user", None), "username", None),
            "content": getattr(tw, "rawContent", getattr(tw, "content", "")),
            "likeCount": getattr(tw, "likeCount", 0),
            "retweetCount": getattr(tw, "retweetCount", 0),
            "replyCount": getattr(tw, "replyCount", 0),
        })
        i += 1
        if i >= int(limit):
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["content"])
    log(f"Slice rows: {len(df)}")
    return df

def scrape_topic(topic: str, lang: str, since_dt: datetime, until_dt: datetime, per_slice_limit: int,
                 slice_hours: int = 12) -> pd.DataFrame:
    """Scrape in time slices; if a slice fails, retry with smaller slices."""
    from snscrape.modules.twitter import TwitterSearchScraper  # ensure import early
    frames = []
    cur = since_dt
    while cur < until_dt:
        nxt = min(until_dt, cur + timedelta(hours=slice_hours))
        qparts = [topic]
        if lang: qparts.append(f"lang:{lang}")
        qparts.append(f"since:{cur.date().isoformat()}")
        qparts.append(f"until:{nxt.date().isoformat()}")
        q = " ".join(qparts)
        try:
            frames.append(run_slice(q, per_slice_limit))
        except Exception as e:
            # If a 12h slice fails, split into 3h slices for this window.
            log(f"Slice {cur}–{nxt} failed: {e}. Splitting into 3h sub-slices.")
            sub_cur = cur
            while sub_cur < nxt:
                sub_nxt = min(nxt, sub_cur + timedelta(hours=3))
                qsub = " ".join([
                    topic,
                    f"lang:{lang}" if lang else "",
                    f"since:{sub_cur.date().isoformat()}",
                    f"until:{sub_nxt.date().isoformat()}",
                ]).strip()
                try:
                    frames.append(run_slice(qsub, max(100, per_slice_limit // 4)))
                except Exception as e2:
                    log(f"Sub-slice {sub_cur}–{sub_nxt} hard failed: {e2}")
                sub_cur = sub_nxt
        cur = nxt
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("date").drop_duplicates(subset=["username", "content"], keep="last")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=str, default="vote chori, stock market")
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--outdir", type=str, default="data")
    ap.add_argument("--slice_hours", type=int, default=12, help="Time-slice size (hours)")
    args = ap.parse_args()

    try:
        import snscrape  # noqa
    except Exception as e:
        log(f"ERROR: snscrape import failed: {e}")
        sys.exit(2)

    since_dt = datetime.utcnow() - timedelta(days=int(args.days))
    until_dt = datetime.utcnow()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    import os
    os.makedirs(args.outdir, exist_ok=True)

    any_written = False
    per_slice_limit = max(100, int(args.limit // max(1, (24 // max(1, args.slice_hours)) * int(args.days))))

    for topic in topics:
        try:
            df = scrape_topic(topic, args.lang or None, since_dt, until_dt, per_slice_limit, slice_hours=args.slice_hours)
        except Exception:
            traceback.print_exc()
            sys.exit(2)

        safe = "_".join(topic.lower().split())
        stamp = datetime.utcnow().strftime("%Y%m%d")
        path_daily  = os.path.join(args.outdir, f"{safe}_{stamp}.csv")
        path_latest = os.path.join(args.outdir, f"latest_{safe}.csv")
        df.to_csv(path_daily, index=False, encoding="utf-8")
        df.to_csv(path_latest, index=False, encoding="utf-8")
        log(f"Wrote {len(df)} rows for '{topic}' -> {path_daily} and {path_latest}")
        any_written = True

    if not any_written:
        log("ERROR: Nothing written.")
        sys.exit(3)

if __name__ == "__main__":
    main()
