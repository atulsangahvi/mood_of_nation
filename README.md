
# Mood of India — Twitter/X Topic Sentiment

A Streamlit app that fetches recent tweets for a topic (e.g., `vote chori`) and estimates **support / oppose / neutral** sentiment.

## Deploy on Streamlit Cloud

1. Create a repo with these files:
   - `mood_of_india_streamlit.py`
   - `requirements.txt`

2. In Streamlit Cloud:
   - New app -> Select your repo and branch
   - Entry point: `mood_of_india_streamlit.py`
   - Deploy

### Cloud-safe defaults (already set)
- Transformer OFF by default (uses VADER to save RAM)
- Prefer TWINT OFF (uses `snscrape` which is more reliable)
- `limit=1000`, `days=2`

You can toggle the transformer on in the sidebar when running locally.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run mood_of_india_streamlit.py
```

### Optional: TWINT
TWINT is brittle; use at your own risk:
```bash
pip install twint-fork
# or:
pip install "git+https://github.com/twintproject/twint.git"
```

## Notes
- `snscrape` uses `place_country:IN` which includes tweets with place metadata tagged as India; some India tweets without place may be missed.
- Sentiment is noisy; for better accuracy, expand the Hindi/Hinglish lexicon or fine-tune a model on Indian political tweets.
