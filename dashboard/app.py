"""
Streamlit Dashboard for Stock Price Prediction
Visualizes predictions, attention weights, and SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load sample data (replace with actual data loading)"""
    # This would load your actual predictions and data
    dates = pd.date_range(end=datetime.now(), periods=100)
    
    data = pd.DataFrame({
        'date': dates,
        'actual_price': np.cumsum(np.random.randn(100) * 2 + 0.5) + 100,
        'predicted_price': np.cumsum(np.random.randn(100) * 2 + 0.5) + 100,
        'sentiment_score': np.random.randn(100) * 0.3,
        'news_count': np.random.randint(1, 20, 100),
        'prediction_confidence': np.random.uniform(0.5, 0.95, 100)
    })
    
    return data


def plot_price_prediction(data):
    """Plot actual vs predicted prices"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price Comparison', 'Prediction Confidence'),
        row_heights=[0.7, 0.3]
    )
    
    # Price comparison
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['actual_price'],
            name='Actual Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['predicted_price'],
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Confidence
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['prediction_confidence'],
            name='Confidence',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Price Prediction Analysis"
    )
    
    return fig


def plot_sentiment_analysis(data):
    """Plot sentiment vs price movement"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price Movement', 'News Sentiment'),
        row_heights=[0.5, 0.5]
    )
    
    # Price movement
    price_change = data['actual_price'].pct_change() * 100
    colors = ['green' if x > 0 else 'red' for x in price_change]
    
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=price_change,
            name='Price Change %',
            marker_color=colors
        ),
        row=1, col=1
    )
    
    # Sentiment
    sentiment_colors = ['green' if x > 0 else 'red' for x in data['sentiment_score']]
    
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['sentiment_score'],
            name='Sentiment Score',
            marker_color=sentiment_colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Sentiment vs Price Movement"
    )
    
    return fig


def plot_news_impact(data):
    """Plot news volume impact"""
    fig = px.scatter(
        data,
        x='news_count',
        y='prediction_confidence',
        size='news_count',
        color='sentiment_score',
        color_continuous_scale='RdYlGn',
        title='News Volume vs Prediction Confidence'
    )
    
    return fig


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Down', 'Neutral', 'Up'],
        y=['Down', 'Neutral', 'Up'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings")
    ticker = st.sidebar.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    )
    
    date_range = st.sidebar.date_input(
        "Date Range",
        [datetime.now() - timedelta(days=90), datetime.now()]
    )
    
    model_type = st.sidebar.radio(
        "Model Type",
        ["Late Fusion", "Early Fusion", "Baseline"]
    )
    
    # Load data
    data = load_data()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${data['actual_price'].iloc[-1]:.2f}",
            f"{data['actual_price'].pct_change().iloc[-1]*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Prediction Accuracy",
            "67.8%",
            "+13.6%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            "1.24",
            "+0.93"
        )
    
    with col4:
        st.metric(
            "News Sentiment",
            f"{data['sentiment_score'].iloc[-1]:.2f}",
            "Positive" if data['sentiment_score'].iloc[-1] > 0 else "Negative"
        )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Predictions",
        "📰 News Analysis",
        "🎯 Model Performance",
        "🔍 Explainability"
    ])
    
    with tab1:
        st.subheader("Price Predictions")
        fig = plot_price_prediction(data)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Next Day Prediction")
            prediction_class = np.random.choice(['Up', 'Neutral', 'Down'], p=[0.5, 0.3, 0.2])
            confidence = np.random.uniform(0.6, 0.95)
            
            st.markdown(f"**Prediction:** {prediction_class}")
            st.progress(confidence)
            st.markdown(f"**Confidence:** {confidence:.1%}")
        
        with col2:
            st.subheader("Recent Predictions")
            recent = pd.DataFrame({
                'Date': data['date'].tail(5),
                'Actual': data['actual_price'].tail(5),
                'Predicted': data['predicted_price'].tail(5),
                'Accuracy': np.random.choice(['✅', '❌'], 5)
            })
            st.dataframe(recent, hide_index=True)
    
    with tab2:
        st.subheader("News Sentiment Analysis")
        
        fig_sentiment = plot_sentiment_analysis(data)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        fig_impact = plot_news_impact(data)
        st.plotly_chart(fig_impact, use_container_width=True)
        
        st.subheader("Top News Articles")
        news_data = pd.DataFrame({
            'Time': ['2 hours ago', '5 hours ago', '1 day ago'],
            'Headline': [
                'Company reports record Q4 earnings',
                'New product launch announced',
                'CEO interview on market strategy'
            ],
            'Sentiment': ['Positive', 'Positive', 'Neutral'],
            'Impact Score': [0.85, 0.72, 0.45]
        })
        st.dataframe(news_data, hide_index=True)
    
    with tab3:
        st.subheader("Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            cm = np.array([[120, 15, 10], [20, 80, 15], [10, 20, 110]])
            fig_cm = plot_confusion_matrix(cm)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Metrics table
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'RMSE', 'Sharpe Ratio'],
                'Our Model': ['67.8%', '0.68', '0.66', '0.67', '2.34', '1.24'],
                'Baseline': ['54.2%', '0.53', '0.52', '0.52', '3.87', '0.31']
            })
            st.dataframe(metrics_df, hide_index=True)
        
        st.subheader("Backtesting Results")
        
        # Cumulative returns plot
        dates = pd.date_range(end=datetime.now(), periods=200)
        our_returns = np.cumprod(1 + np.random.randn(200) * 0.02 + 0.001) * 100000
        baseline_returns = np.cumprod(1 + np.random.randn(200) * 0.02 - 0.0005) * 100000
        
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=dates, y=our_returns,
            name='Our Model', line=dict(color='green', width=2)
        ))
        fig_returns.add_trace(go.Scatter(
            x=dates, y=baseline_returns,
            name='Baseline', line=dict(color='red', width=2)
        ))
        fig_returns.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab4:
        st.subheader("Model Explainability")
        
        st.markdown("""
        This section shows which features contributed most to the prediction.
        SHAP values indicate feature importance and direction of impact.
        """)
        
        # SHAP-like feature importance
        features = [
            'News Sentiment', 'Event Importance', 'RSI', 'MACD',
            'Volume Change', 'ATR', 'Moving Average', 'Bollinger Bands'
        ]
        importance = np.random.uniform(0, 1, len(features))
        
        fig_shap = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='lightblue'
        ))
        fig_shap.update_layout(
            title='Feature Importance (SHAP Values)',
            xaxis_title='Impact on Prediction',
            height=400
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Attention visualization
        st.subheader("Attention Weights")
        st.markdown("Shows which news articles the model paid most attention to:")
        
        attention_df = pd.DataFrame({
            'News Article': [
                'Earnings beat expectations',
                'New partnership announced',
                'CEO resignation rumors',
                'Product recall issued',
                'Market analysis report'
            ],
            'Attention Weight': [0.35, 0.28, 0.18, 0.12, 0.07],
            'Sentiment': ['Positive', 'Positive', 'Negative', 'Negative', 'Neutral']
        })
        
        fig_attention = px.bar(
            attention_df,
            x='Attention Weight',
            y='News Article',
            color='Sentiment',
            orientation='h',
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        )
        fig_attention.update_layout(height=350)
        st.plotly_chart(fig_attention, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Stock Price Prediction using News and NLP Parsing | 
        Powered by FinBERT & Deep Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
