from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

from hn_ingestor import HNIngestor, HNIngestorError
from insight_engine import InsightEngine, InsightEngineError

st.set_page_config(
    page_title="Market Gap Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    
    .title-container {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .title-container h1 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    .negativity-card {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    
    .negativity-value {
        font-size: 3rem;
        font-weight: bold;
    }
    
    .business-value-box {
        background: linear-gradient(135deg, #6366F115 0%, #8B5CF615 100%);
        border: 1px solid #6366F140;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    if "df_comments" not in st.session_state:
        st.session_state.df_comments = None
    if "df_analyzed" not in st.session_state:
        st.session_state.df_analyzed = None
    if "pain_points" not in st.session_state:
        st.session_state.pain_points = []
    if "negativity_index" not in st.session_state:
        st.session_state.negativity_index = 0.0
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False


def load_insight_engine() -> InsightEngine:
    return InsightEngine()


def create_complaints_chart(pain_points: List[Tuple[str, float]]) -> go.Figure:
    if not pain_points:
        fig = go.Figure()
        fig.add_annotation(
            text="No complaints extracted yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    top_points = pain_points[:5]
    bigrams = [p[0] for p in top_points]
    scores = [p[1] for p in top_points]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=bigrams,
        orientation='h',
        marker=dict(
            color=['#6366F1', '#8B5CF6', '#A78BFA', '#C4B5FD', '#DDD6FE'],
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="🔥 Top 5 Market Gap Signals (Pain Points)",
        xaxis_title="TF-IDF Score",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(autorange="reversed"),
        height=300
    )
    
    return fig


def display_gap_candidates(df: pd.DataFrame) -> None:
    negative_df = df[df["sentiment_label"] == "Negative"].copy()
    
    if negative_df.empty:
        st.info("No negative comments found - this community seems happy! 🎉")
        return
    
    negative_df = negative_df.sort_values("sentiment_confidence", ascending=False)
    
    st.dataframe(
        negative_df[[
            "story_title", "comment_text", "sentiment_confidence"
        ]].rename(columns={
            "story_title": "Story",
            "comment_text": "Comment",
            "sentiment_confidence": "Confidence"
        }),
        use_container_width=True,
        height=400
    )


def clean_html_for_display(text: str) -> str:
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    text = text.replace('&#x27;', "'").replace('&quot;', '"')
    return text.strip()


def main() -> None:
    init_session_state()
    
    st.markdown("""
    <div class="title-container">
        <h1>🎯 Market Gap Detector</h1>
        <p>Discover unmet needs and market opportunities through community sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📚 **How It Works**", expanded=False):
        st.markdown("""
        <div class="business-value-box">
        
        ### Discover Market Opportunities
        
        **Negative sentiment in online discussions reveals:**
        - Frustrations with existing tools
        - Unmet needs in workflows
        - Emerging pain points in the ecosystem
        
        #### Key Metrics:
        
        - **Negativity Index**: Higher % = more frustration = bigger market gaps
        - **Top Pain Points**: Specific phrases reveal *what* people struggle with
        - **Gap Candidates**: Raw comments to read for full context
        
        #### How to Use:
        
        1. Configure the number of stories and comments to analyze
        2. Click **Fetch & Analyze** to get latest discussions
        3. Review the **Negativity Index** for overall sentiment
        4. Explore **Top Pain Points** for specific opportunities
        5. Read **Gap Candidates** for validation
        
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        num_stories = st.slider(
            "Number of Stories",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of top stories to analyze"
        )
        
        comments_per_story = st.slider(
            "Comments per Story",
            min_value=10,
            max_value=50,
            value=20,
            help="Comments to fetch per story"
        )
        
        st.markdown("---")
        
        fetch_button = st.button(
            "🚀 Fetch & Analyze",
            use_container_width=True,
            type="primary"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Features")
        st.markdown("""
        - **Sentiment Analysis** using NLP
        - **Pain Point Extraction** via TF-IDF
        - **Real-time Insights** with concurrent fetching
        - **Exportable Reports** in CSV format
        """)
    
    if fetch_button:
        try:
            with st.spinner("🧠 Loading NLP models..."):
                engine = load_insight_engine()
            
            with st.spinner(f"📥 Fetching {num_stories} stories × {comments_per_story} comments..."):
                ingestor = HNIngestor(max_workers=20)
                df_comments = ingestor.fetch_top_stories_with_comments(
                    num_stories=num_stories,
                    comments_per_story=comments_per_story
                )
                ingestor.close()
                st.session_state.df_comments = df_comments
            
            if df_comments.empty:
                st.error("❌ No comments fetched. Data source might be unavailable.")
                return
            
            st.success(f"✅ Fetched {len(df_comments)} comments from {df_comments['story_title'].nunique()} stories")
            
            with st.spinner("🔬 Analyzing sentiment..."):
                df_analyzed = engine.analyze_sentiment(df_comments, text_column="comment_text")
                st.session_state.df_analyzed = df_analyzed
            
            negativity_index = engine.get_negativity_index(df_analyzed)
            st.session_state.negativity_index = negativity_index
            
            with st.spinner("🔍 Extracting pain points..."):
                pain_points = engine.extract_pain_points(df_analyzed, top_n=10)
                st.session_state.pain_points = pain_points
            
            st.session_state.analysis_complete = True
            st.success("✅ Analysis complete!")
            
        except HNIngestorError as e:
            st.error(f"❌ Data Fetch Error: {e}")
            return
        except InsightEngineError as e:
            st.error(f"❌ NLP Error: {e}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return
    
    if st.session_state.analysis_complete and st.session_state.df_analyzed is not None:
        df = st.session_state.df_analyzed
        pain_points = st.session_state.pain_points
        negativity_index = st.session_state.negativity_index
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Total Comments",
                value=f"{len(df):,}"
            )
        
        with col2:
            st.metric(
                label="📰 Stories Analyzed",
                value=f"{df['story_title'].nunique()}"
            )
        
        with col3:
            color = "🔴" if negativity_index > 30 else "🟡" if negativity_index > 15 else "🟢"
            st.metric(
                label=f"{color} Negativity Index",
                value=f"{negativity_index}%",
                help="Percentage of comments with negative sentiment"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_complaints = create_complaints_chart(pain_points)
            st.plotly_chart(fig_complaints, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Sentiment Distribution")
            sentiment_counts = df["sentiment_label"].value_counts()
            
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    "Negative": "#e74c3c",
                    "Neutral": "#95a5a6",
                    "Positive": "#27ae60"
                }
            )
            fig_pie.update_layout(
                showlegend=True,
                height=250,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Gap Candidates (Negative Comments)")
        st.caption("These are the actual complaints - read them to understand the pain points in detail.")
        
        display_df = df.copy()
        display_df["comment_text"] = display_df["comment_text"].apply(clean_html_for_display)
        
        display_gap_candidates(display_df)
        
        st.markdown("---")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Analysis (CSV)",
            data=csv,
            file_name="market_gap_analysis.csv",
            mime="text/csv"
        )
    
    else:
        st.info("👈 Click **Fetch & Analyze** in the sidebar to discover market gaps!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📈 Negativity Index")
            st.markdown("See the percentage of negative comments across discussions.")
        
        with col2:
            st.markdown("### 🔥 Top Pain Points")
            st.markdown("Discover the most common frustrations in the community.")
        
        with col3:
            st.markdown("### 🎯 Gap Candidates")
            st.markdown("Read actual negative comments to understand opportunities.")


if __name__ == "__main__":
    main()
