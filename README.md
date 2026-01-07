# 🎯 Market Gap Detector

> **Discover unmet needs and market opportunities through community sentiment analysis**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Overview

Market Gap Detector is a powerful NLP-powered tool that analyzes online discussions to identify market opportunities. By detecting negative sentiment and extracting pain points from community conversations, it helps entrepreneurs and product teams discover unmet needs.

### 💡 Core Concept

| Sentiment | What It Means |
|-----------|---------------|
| 🔴 **Negative** | **Market Gap** — Frustration = Business Opportunity |
| ⚪ Neutral | General discussion or information |
| 🟢 Positive | Existing solutions are working well |

---

## ✨ Features

- ** Sentiment Analysis** — Uses HuggingFace transformers with rule-based fallback
- ** Pain Point Extraction** — TF-IDF weighted bigram analysis on negative comments
- ** Concurrent Fetching** — 10-20x faster data ingestion with ThreadPoolExecutor
- ** Interactive Dashboard** — Real-time visualizations with Plotly
- ** Export Results** — Download analysis as CSV for further processing

---

##  Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/market-gap-detector.git
cd market-gap-detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run hn_app.py
```

The app will open at `http://localhost:8501`

---

##  Screenshots

| Dashboard | Analysis Results |
|-----------|------------------|
| Configure stories & comments to analyze | View negativity index, pain points, and sentiment distribution |

---

##  Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Source    │────▶│  Data Ingestor  │────▶│  InsightEngine  │
│                 │     │  (Concurrent)   │     │  (NLP Pipeline) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │  Streamlit App  │
                                               │  (Dashboard)    │
                                               └─────────────────┘
```

---

##  Project Structure

```
market-gap-detector/
├── hn_app.py           # Streamlit dashboard (main entry point)
├── hn_ingestor.py      # Data ingestion module (concurrent fetching)
├── insight_engine.py   # NLP pipeline (sentiment + pain points)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

##  Usage

### As a Streamlit App
Simply run `streamlit run hn_app.py` and use the interactive dashboard.

### As Python Modules

```python
from hn_ingestor import HNIngestor
from insight_engine import InsightEngine

# Fetch data
ingestor = HNIngestor(max_workers=20)
df = ingestor.fetch_top_stories_with_comments(num_stories=10, comments_per_story=20)
ingestor.close()

# Analyze sentiment
engine = InsightEngine()
df_analyzed = engine.analyze_sentiment(df, text_column="comment_text")

# Get metrics
negativity = engine.get_negativity_index(df_analyzed)
pain_points = engine.extract_pain_points(df_analyzed, top_n=10)

print(f"Negativity Index: {negativity}%")
print(f"Top Pain Points: {pain_points[:5]}")
```

---

##  Use Cases

1. **Startup Ideation** — Find pain points in online communities
2. **Product Development** — Discover feature gaps in existing tools
3. **Market Research** — Understand community sentiment trends
4. **Competitive Analysis** — See what users dislike about current solutions

---

##  Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_stories` | 10 | Number of top stories to analyze |
| `comments_per_story` | 20 | Comments to fetch per story |
| `max_workers` | 20 | Concurrent threads for fetching |

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
