"""Download two public, human-labelled tweet datasets for measuring accuracy. They are not stored in this repository.

  airline: 14,640 tweets to US airlines, Feb 2015, labelled negative, neutral or positive, with a reason for negatives.
           CrowdFlower "Twitter US Airline Sentiment", copied from a public GitHub mirror.
  sxsw:    9,093 tweets from SXSW 2011 about Apple and Google products, labelled with the emotion directed at a brand.
           CrowdFlower "Brands and Product Emotions", copied from a public GitHub mirror.
"""
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "benchmarks" / "data"
SOURCES = {
    "airline.csv": "https://raw.githubusercontent.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/master/Tweets.csv",
    "sxsw.csv": "https://raw.githubusercontent.com/lukas/ml-class/master/videos/text-classifier/tweets.csv",
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = DATA / name
        if path.exists():
            print(f"Already have {path}")
            continue
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"Saved {path} ({len(r.content):,} bytes)")


if __name__ == "__main__":
    main()
