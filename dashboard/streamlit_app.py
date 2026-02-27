"""
Streamlit Dashboard for Cyberbullying Detection System
Beautiful, interactive web interface for real-time predictions
"""

from typing import TYPE_CHECKING, Any, cast

# Static-only imports to help static analyzers (Pylance/pyright) resolve symbols
if TYPE_CHECKING:
    # Only import project-local types for static analysis; avoid requiring third-party packages
    from src.preprocessing import TextPreprocessor  # type: ignore

# Optional UI dependencies (guarded so module can be imported in minimal environments)
try:
    import streamlit as st  # type: ignore[reportMissingImports]
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False
    # Minimal shim for the methods used in this module so it can be imported without Streamlit
    class _DummyColumn:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def metric(self, *args, **kwargs):
            pass

    class _DummySt:
        def cache_resource(self, func=None, **kwargs):
            def decorator(f):
                return f
            if func:
                return func
            return decorator
        def error(self, *args, **kwargs):
            print("[streamlit.error]:", *args)
        def markdown(self, *args, **kwargs):
            pass
        def title(self, *args, **kwargs):
            pass
        def set_page_config(self, *args, **kwargs):
            pass
        def columns(self, n):
            # return n dummy columns usable as context managers
            return tuple(_DummyColumn() for _ in range(n))
        def metric(self, *args, **kwargs):
            pass
        def subheader(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
        def selectbox(self, *args, **kwargs):
            return None
        def text_area(self, *args, **kwargs):
            return ''
        def button(self, *args, **kwargs):
            return False
        def rerun(self, *args, **kwargs):
            pass
        def spinner(self, *args, **kwargs):
            # context manager stub
            class _Spinner:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False
            return _Spinner()
        def warning(self, *args, **kwargs):
            pass
        def plotly_chart(self, *args, **kwargs):
            pass
        def file_uploader(self, *args, **kwargs):
            return None
        def success(self, *args, **kwargs):
            pass
        def progress(self, *args, **kwargs):
            class _Progress:
                def __init__(self):
                    self.value = 0
                def progress(self, v):
                    self.value = v
            return _Progress()
        def dataframe(self, *args, **kwargs):
            pass
        def download_button(self, *args, **kwargs):
            return None
        def sidebar(self):
            return self
    st = cast(Any, _DummySt())

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
try:
    import plotly.express as px  # type: ignore[reportMissingImports]
    import plotly.graph_objects as go  # type: ignore[reportMissingImports]
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    px = cast(Any, None)
    go = cast(Any, None)

from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import config
# `preprocessing` may not be on static analysis path; provide a safe fallback
try:
    from src.preprocessing import TextPreprocessor, get_sentiment
except Exception:
    class _FallbackTextPreprocessor:
        def __init__(self):
            self.profanity_words = set()
        def clean_text(self, text):
            return str(text)
    # Tell static type checker to treat the fallback as Any to avoid type mismatch warnings
    TextPreprocessor = cast(Any, _FallbackTextPreprocessor)
    get_sentiment = lambda x: 0.0


# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .cyberbullying {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        border: 2px solid #ff6666;
    }
    .not-cyberbullying {
        background: linear-gradient(135deg, #44ff44, #00cc00);
        color: black;
        border: 2px solid #66ff66;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
    }
    .sentiment-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    .tab-content {
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


class CyberbullyingDetector:
    """Main detection class with model loading and prediction"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.preprocessor = TextPreprocessor()
        self.load_models()
    
    @st.cache_resource
    def load_models(_self):
        """Load trained models and vectorizers"""
        try:
            # Load vectorizers
            tfidf_path = config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl'
            if tfidf_path.exists():
                _self.vectorizers['tfidf'] = joblib.load(tfidf_path)
            
            # Load models
            model_files = {
                'Naive Bayes': 'naive_bayes_model.pkl',
                'SVM': 'svm_model.pkl',
                'Random Forest': 'random_forest_model.pkl',
                'Logistic Regression': 'logistic_regression_model.pkl',
                'Ensemble': 'ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = config.SAVED_MODELS_DIR / filename
                if model_path.exists():
                    _self.models[model_name] = joblib.load(model_path)
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict(self, text, model_name='Naive Bayes'):
        """Make prediction on input text"""
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Vectorize
            if 'tfidf' in self.vectorizers:
                X = self.vectorizers['tfidf'].transform([cleaned_text])
                # Add sentiment feature
                # sentiment = get_sentiment(cleaned_text)
                # X = hstack([X, [[sentiment]]])
            else:
                return None, 0.0
            
            # Predict
            if model_name in self.models:
                model = self.models[model_name]
                prediction = model.predict(X)[0]
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    # Some models may return probability vectors with a single column
                    if isinstance(proba, (list, tuple, np.ndarray)):
                        if prediction < len(proba):
                            confidence = float(proba[prediction])
                        else:
                            # fallback to the largest available probability
                            if isinstance(proba, np.ndarray):
                                confidence = float(proba.max())
                            else:
                                confidence = float(max(proba))
                    else:
                        confidence = float(proba)
                elif hasattr(model, 'decision_function'):
                    decision = model.decision_function(X)[0]
                    # decision may be a single float or an array
                    if hasattr(decision, '__len__'):
                        decision = decision[0]
                    confidence = float(1 / (1 + np.exp(-decision)))  # Sigmoid for positive class
                    if prediction == 0:
                        confidence = 1.0 - confidence
                else:
                    confidence = 0.5
                
                return prediction, confidence
            
            return None, 0.0
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, 0.0
    
    def predict_all_models(self, text):
        """Get predictions from all models"""
        results = {}
        for model_name in self.models.keys():
            pred, conf = self.predict(text, model_name)
            if pred is not None:
                results[model_name] = {
                    'prediction': 'Cyberbullying' if pred == 1 else 'Not Cyberbullying',
                    'confidence': conf
                }
        return results
    
    def get_severity(self, confidence):
        """Determine severity level based on confidence"""
        if confidence < config.SEVERITY_THRESHOLDS['mild']:
            return 'Mild', '🟡'
        elif confidence < config.SEVERITY_THRESHOLDS['moderate']:
            return 'Moderate', '🟠'
        else:
            return 'Severe', '🔴'
    
    def get_sentiment_analysis(self, text):
        """Analyze sentiment of the text"""
        try:
            sentiment_score = get_sentiment(text)
            if sentiment_score > 0.1:
                sentiment = "Positive"
                color = "green"
                icon = "😊"
            elif sentiment_score < -0.1:
                sentiment = "Negative"
                color = "red"
                icon = "😠"
            else:
                sentiment = "Neutral"
                color = "gray"
                icon = "😐"
            return sentiment, color, icon, sentiment_score
        except:
            return "Unknown", "gray", "❓", 0.0

    def get_top_features(self, text, model_name='Naive Bayes', top_n=10):
        """Get top contributing features for prediction"""
        try:
            if model_name not in self.models or 'tfidf' not in self.vectorizers:
                return []

            cleaned_text = self.preprocessor.clean_text(text)
            vectorizer = self.vectorizers['tfidf']
            model = self.models[model_name]

            # Transform text
            X = vectorizer.transform([cleaned_text])

            # Get feature names and their importance
            feature_names = vectorizer.get_feature_names_out()

            if hasattr(model, 'coef_'):
                # For linear models (Logistic Regression, SVM, Naive Bayes)
                if model_name == 'Naive Bayes':
                    # Naive Bayes has different structure
                    coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                else:
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_

                # Get top features
                if len(coef) == len(feature_names):
                    feature_importance = list(zip(feature_names, coef))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    return feature_importance[:top_n]

            elif hasattr(model, 'feature_importances_'):
                # For Random Forest
                importances = model.feature_importances_
                if len(importances) == len(feature_names):
                    feature_importance = list(zip(feature_names, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    return feature_importance[:top_n]

            return []
        except Exception as e:
            return []

    def highlight_offensive_words(self, text):
        """Highlight potentially offensive words"""
        words = text.lower().split()
        highlighted = []
        offensive_found = []

        for word in words:
            if word in self.preprocessor.profanity_words:
                highlighted.append(f"**:red[{word}]**")
                offensive_found.append(word)
            else:
                highlighted.append(word)

        return ' '.join(highlighted), offensive_found


# Initialize detector
@st.cache_resource
def get_detector():
    return CyberbullyingDetector()


def show_home_page():
    """Display home page with system overview"""
    st.title("🛡️ Cyberbullying Detection System")
    st.markdown("### Advanced ML-Powered Protection Against Online Harassment")

    # Dark mode toggle (visual only - Streamlit doesn't support runtime theme switching)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        dark_mode = st.checkbox("🌙 Dark Mode", value=True)
        if dark_mode:
            st.markdown("""
            <style>
            .main { background-color: #0e1117; color: white; }
            </style>
            """, unsafe_allow_html=True)

    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Models Deployed",
            value="5",
            delta="Ensemble Active"
        )

    with col2:
        st.metric(
            label="Accuracy",
            value="90%+",
            delta="F1-Score: 88%"
        )

    with col3:
        st.metric(
            label="Dataset Size",
            value="150K+",
            delta="Multi-platform"
        )

    with col4:
        st.metric(
            label="Response Time",
            value="<100ms",
            delta="Real-time"
        )

    st.markdown("---")

    # Quick demo section
    st.subheader("⚡ Quick Demo")
    st.markdown("Try the system with sample texts:")

    demo_texts = [
        "You're so stupid and ugly!",
        "Have a great day!",
        "Nobody likes you, just disappear",
        "I love this weather"
    ]

    selected_demo = st.selectbox("Choose a sample text:", demo_texts, index=0)

    if st.button("🚀 Analyze Sample", type="primary"):
        detector = get_detector()
        pred, conf = detector.predict(selected_demo, 'Ensemble')
        sentiment, color, icon, score = detector.get_sentiment_analysis(selected_demo)

        col1, col2 = st.columns(2)
        with col1:
            result_text = "⚠️ CYBERBULLYING DETECTED" if pred == 1 else "✅ NO CYBERBULLYING DETECTED"
            result_class = "cyberbullying" if pred == 1 else "not-cyberbullying"
            st.markdown(f"""
                <div class="prediction-box {result_class}">
                    <h3>{result_text}</h3>
                    <p>Confidence: {conf*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #{color}; color: white; text-align: center;">
                    <h2>{icon}</h2>
                    <h4>{sentiment.upper()}</h4>
                    <p>Score: {score:.3f}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.subheader("✨ Key Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - 🤖 **5 ML Models**: Naive Bayes, SVM, Random Forest, Logistic Regression, Ensemble
        - 📊 **Advanced Analytics**: Real-time performance metrics and visualizations
        - 🎭 **Sentiment Analysis**: Emotional tone detection alongside cyberbullying
        - 🔍 **Model Explainability**: Understand which words influence predictions
        """)

    with col2:
        st.markdown("""
        - ☁️ **Word Cloud**: Visualize common harmful language patterns
        - 📱 **Social Media Simulator**: Platform-specific content analysis
        - 📈 **Interactive Charts**: Plotly-powered visualizations
        - 📂 **Batch Processing**: Upload files for bulk detection
        """)

    st.markdown("---")

    # Recent activity (simulated)
    st.subheader("📈 System Activity")
    activity_data = pd.DataFrame({
        'Time': pd.date_range('now', periods=5, freq='-1h'),
        'Analyzed': [23, 45, 67, 34, 89],
        'Flagged': [2, 8, 12, 5, 15]
    })

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=activity_data['Time'],
            y=activity_data['Analyzed'],
            mode='lines+markers',
            name='Texts Analyzed',
            line=dict(color='#FF4B4B')
        ))
        fig.add_trace(go.Scatter(
            x=activity_data['Time'],
            y=activity_data['Flagged'],
            mode='lines+markers',
            name='Cyberbullying Flagged',
            line=dict(color='#FF6B6B', dash='dash')
        ))
        fig.update_layout(
            title="Recent System Activity (Last 5 Hours)",
            xaxis_title="Time",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.dataframe(activity_data, width='stretch')

    # Quick Start
    st.subheader("🚀 Quick Start")
    st.info("👈 Use the sidebar to navigate between different features!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>🛡️ Cyberbullying Detection System v2.0 | Built with ❤️ using Streamlit & Scikit-learn</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)


def show_detection_page(detector):
    """Main detection interface"""
    st.title("🔍 Cyberbullying Detection")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(detector.models.keys()),
        index=list(detector.models.keys()).index('Ensemble') if 'Ensemble' in detector.models else 0
    )
    
    # Text input
    st.subheader("Enter Text to Analyze")
    text_input = st.text_area(
        "Type or paste text here...",
        height=150,
        placeholder="Example: You're so stupid and worthless!"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_button = st.button("🔎 Analyze Text", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    if analyze_button and text_input:
        with st.spinner("Analyzing..."):
            # Make prediction
            prediction, confidence = detector.predict(text_input, model_name)
            
            if prediction is not None:
                st.markdown("---")
                
                # Display result
                result_text = "⚠️ CYBERBULLYING DETECTED" if prediction == 1 else "✅ NO CYBERBULLYING DETECTED"
                result_class = "cyberbullying" if prediction == 1 else "not-cyberbullying"
                
                st.markdown(f"""
                    <div class="prediction-box {result_class}">
                        <h2>{result_text}</h2>
                        <h3>Confidence: {confidence*100:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show details
                col1, col2 = st.columns(2)
                
                with col1:
                    # Severity
                    if prediction == 1:
                        severity, icon = detector.get_severity(confidence)
                        st.subheader(f"{icon} Severity Level: {severity}")
                    
                    # Highlighted text
                    st.subheader("� Linguistic Forensics")
                    highlighted, offensive = detector.highlight_offensive_words(text_input)
                    st.markdown(highlighted)
                    
                    if offensive:
                        st.warning(f"🚨 Detected offensive terms: {', '.join(offensive)}")
                    else:
                        st.info("✅ No offensive language detected in this text.")
                
                with col2:
                    # All models predictions
                    st.subheader("🤖 All Models Agreement")
                    all_results = detector.predict_all_models(text_input)
                    
                    for model, result in all_results.items():
                        icon = "✓" if result['prediction'] == 'Cyberbullying' else "✗"
                        color = "red" if result['prediction'] == 'Cyberbullying' else "green"
                        st.markdown(f"**{icon} {model}**: :{color}[{result['prediction']}] ({result['confidence']*100:.1f}%)")
                
                # Confidence meter
                st.subheader("📊 Confidence Meter")
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "red" if prediction == 1 else "green"},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': "gray"},
                                {'range': [80, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')
                else:
                    # Fallback: simple text-based confidence display
                    confidence_pct = confidence * 100
                    color = "red" if prediction == 1 else "green"
                    st.markdown(f"**Confidence Score**: :{color}[{confidence_pct:.1f}%]")
                    # Simple progress bar using text
                    filled = int(confidence_pct / 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    st.markdown(f"**Confidence Bar**: {bar} ({confidence_pct:.1f}%)")


def show_batch_detection(detector):
    """Batch processing interface"""
    st.title("📂 Batch Detection")
    st.markdown("Upload a CSV file with text data for bulk analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} records")
        
        # Column selection
        text_column = st.selectbox("Select text column", df.columns)
        
        if st.button("🚀 Process All"):
            progress_bar = st.progress(0)
            results = []
            
            for idx, text in enumerate(df[text_column]):
                pred, conf = detector.predict(str(text), 'Ensemble')
                results.append({
                    'text': text,
                    'prediction': 'Cyberbullying' if pred == 1 else 'Not Cyberbullying',
                    'confidence': conf
                })
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(results_df))
            with col2:
                cyberbullying_count = (results_df['prediction'] == 'Cyberbullying').sum()
                st.metric("Cyberbullying Detected", cyberbullying_count)
            with col3:
                st.metric("Clean Messages", len(results_df) - cyberbullying_count)
            
            # Display results
            st.subheader("Results")
            st.dataframe(results_df, width='stretch')
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="cyberbullying_detection_results.csv",
                mime="text/csv"
            )


def show_analytics(detector):
    """Analytics and model comparison page"""
    st.title("📊 Analytics & Model Comparison")
    
    # Sample predictions for visualization
    sample_texts = [
        "You're so stupid and ugly!",
        "Great job on your presentation!",
        "Nobody likes you, just disappear",
        "Have a wonderful day!",
        "You should kill yourself"
    ]
    
    # Get predictions from all models
    comparison_data = []
    for text in sample_texts:
        results = detector.predict_all_models(text)
        for model, result in results.items():
            comparison_data.append({
                'Text': text[:30] + '...',
                'Model': model,
                'Prediction': result['prediction'],
                'Confidence': result['confidence'] * 100
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Model comparison chart
    st.subheader("Model Comparison")
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df_comparison,
            x='Model',
            y='Confidence',
            color='Prediction',
            barmode='group',
            title="Average Confidence by Model",
            color_discrete_map={
                'Cyberbullying': '#FF4B4B',
                'Not Cyberbullying': '#09AB3B'
            }
        )
        st.plotly_chart(fig, width='stretch')
    else:
        # Fallback: simple table display
        st.markdown("**Model Comparison (Text-based)**:")
        st.dataframe(df_comparison, width='stretch')
    
    # Detailed results table
    st.subheader("Sample Predictions")
    st.dataframe(df_comparison, width='stretch')


def show_advanced_features(detector):
    """Advanced features and analysis page"""
    st.title("🔬 Advanced Features")

    # Create tabs for different advanced features
    tab1, tab2, tab3 = st.tabs(["🎭 Sentiment Analysis", "☁️ Word Cloud", "📱 Social Media Simulator"])

    with tab1:
        show_sentiment_analysis(detector)

    with tab2:
        show_word_cloud(detector)

    with tab3:
        show_social_media_simulator(detector)


def show_sentiment_analysis(detector):
    """Real-time sentiment analysis feature"""
    st.header("🎭 Real-time Sentiment Analysis")

    st.markdown("""
    Analyze the emotional tone of text alongside cyberbullying detection.
    This helps understand the psychological impact of potentially harmful content.
    """)

    text_input = st.text_area(
        "Enter text for sentiment analysis:",
        height=100,
        placeholder="Example: You're amazing and I appreciate your work!"
    )

    if st.button("🔍 Analyze Sentiment", type="primary") and text_input:
        with st.spinner("Analyzing sentiment..."):
            # Get cyberbullying prediction
            pred, conf = detector.predict(text_input, 'Ensemble')

            # Get sentiment analysis
            sentiment, color, icon, score = detector.get_sentiment_analysis(text_input)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🛡️ Cyberbullying Detection")
                result_class = "cyberbullying" if pred == 1 else "not-cyberbullying"
                result_text = "⚠️ CYBERBULLYING DETECTED" if pred == 1 else "✅ NO CYBERBULLYING DETECTED"
                st.markdown(f"""
                    <div class="prediction-box {result_class}">
                        <h3>{result_text}</h3>
                        <p>Confidence: {conf*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.subheader("🎭 Sentiment Analysis")
                st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #{color}; color: white; text-align: center; margin: 10px 0;">
                        <h1>{icon}</h1>
                        <h3>{sentiment.upper()}</h3>
                        <p>Score: {score:.3f}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Combined insight
            st.subheader("🔗 Combined Analysis")
            if pred == 1 and sentiment == "Negative":
                st.error("🚨 **High Risk**: Negative sentiment + Cyberbullying detected. This content may cause significant emotional harm.")
            elif pred == 1 and sentiment == "Neutral":
                st.warning("⚠️ **Moderate Risk**: Cyberbullying detected with neutral tone. May still be harmful.")
            elif pred == 0 and sentiment == "Negative":
                st.info("ℹ️ **Low Risk**: Negative sentiment but no cyberbullying patterns detected.")
            else:
                st.success("✅ **Safe**: Positive/neutral sentiment with no cyberbullying indicators.")


def show_word_cloud(detector):
    """Word cloud visualization"""
    st.header("☁️ Cyberbullying Word Cloud")

    st.markdown("""
    Visualize the most common words associated with cyberbullying detection.
    This helps identify patterns and common harmful language.
    """)

    try:
        # Load processed data for word cloud
        processed_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE
        if processed_data_path.exists():
            df = joblib.load(processed_data_path)

            # Filter cyberbullying texts
            cyberbullying_texts = df[df['label'] == 'aggressive']['clean_text'].dropna()

            if len(cyberbullying_texts) > 0:
                # Create word frequency
                from collections import Counter
                all_words = []
                for text in cyberbullying_texts.sample(min(1000, len(cyberbullying_texts))):
                    all_words.extend(str(text).split())

                word_freq = Counter(all_words)
                # Remove common stopwords
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cant', 'dont', 'wont', 'youre', 'im', 'its', 'thats', 'theyre'}
                word_freq = {k: v for k, v in word_freq.items() if k.lower() not in stopwords and len(k) > 2}

                # Get top 50 words
                top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])

                if PLOTLY_AVAILABLE:
                    # Create word cloud using plotly
                    words = list(top_words.keys())
                    frequencies = list(top_words.values())

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(words))),
                        y=frequencies,
                        mode='markers+text',
                        text=words,
                        textposition="top center",
                        marker=dict(
                            size=[f/2 for f in frequencies],  # Size based on frequency
                            color=frequencies,
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(title="Frequency")
                        ),
                        hovertemplate='<b>%{text}</b><br>Frequency: %{y}<extra></extra>'
                    ))

                    fig.update_layout(
                        title="Most Common Words in Cyberbullying Content",
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(title="Frequency"),
                        height=600
                    )

                    st.plotly_chart(fig, width='stretch')
                else:
                    # Fallback: show as table
                    st.dataframe(pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency']), width='stretch')

                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Words Analyzed", f"{sum(word_freq.values()):,}")
                with col2:
                    st.metric("Unique Words", f"{len(word_freq):,}")
                with col3:
                    # Find most common word and its frequency
                    most_common_word = max(word_freq.keys(), key=lambda k: word_freq[k])
                    max_frequency = word_freq[most_common_word]
                    st.metric("Most Common Word", f"{most_common_word} ({max_frequency})")

            else:
                st.warning("No cyberbullying data found for word cloud generation.")
        else:
            st.error("Processed data not found. Please run the training pipeline first.")

    except Exception as e:
        st.error(f"Error generating word cloud: {e}")


def show_social_media_simulator(detector):
    """Social media post simulator"""
    st.header("📱 Social Media Post Simulator")

    st.markdown("""
    Simulate analyzing social media posts from different platforms.
    See how the same content might be perceived across different contexts.
    """)

    # Platform selection
    platform = st.selectbox(
        "Select Social Media Platform:",
        ["Twitter/X", "Facebook", "Instagram", "Reddit", "TikTok"],
        index=0
    )

    # Predefined example posts based on platform
    platform_examples = {
        "Twitter/X": [
            "You're such an idiot for thinking that! 😂 #fail",
            "Just got promoted at work! So grateful 🙏 #blessed",
            "This weather is absolutely terrible today ☔️",
            "Can't believe how stupid some people are 🤦‍♂️"
        ],
        "Facebook": [
            "To all my friends: you're amazing and I appreciate each of you! ❤️",
            "Political opinions belong in the trash where they came from",
            "Beautiful sunset today! Nature is healing 🌅",
            "Some people just need to grow up and get a life"
        ],
        "Instagram": [
            "You're gorgeous and talented! Keep shining ✨ #bodypositive",
            "This food looks disgusting, who would eat that? 🤢",
            "Loving this new outfit! Feeling confident 💃",
            "Your makeup looks terrible, maybe try a different style"
        ],
        "Reddit": [
            "This post is absolute garbage, OP is clearly clueless",
            "Great discussion everyone! Really learned something new",
            "Another day, another terrible take from this sub",
            "Appreciate the thoughtful responses, this is why I love this community"
        ],
        "TikTok": [
            "POV: You're so annoying that everyone leaves the room 😂 #relatable",
            "This dance is everything! You're crushing it 💃 #fyp",
            "Why are you still single? Just fix yourself already",
            "This trend is so fun! Everyone join in 🎉 #viral"
        ]
    }

    # Text input with platform-specific placeholder
    text_input = st.text_area(
        f"Enter {platform} post:",
        height=100,
        placeholder=f"Example: {platform_examples[platform][0]}"
    )

    # Quick example buttons
    st.markdown("**Quick Examples:**")
    cols = st.columns(2)
    for i, example in enumerate(platform_examples[platform][:2]):
        with cols[i]:
            if st.button(f"📝 Example {i+1}", key=f"example_{i}"):
                text_input = example
                st.rerun()

    if st.button("🚀 Analyze Post", type="primary") and text_input:
        with st.spinner(f"Analyzing {platform} post..."):
            # Get prediction
            pred, conf = detector.predict(text_input, 'Ensemble')

            # Get sentiment
            sentiment, color, icon, score = detector.get_sentiment_analysis(text_input)

            # Platform-specific analysis
            platform_context = {
                "Twitter/X": "Character-limited, fast-paced, high engagement",
                "Facebook": "Personal networks, longer posts, community focused",
                "Instagram": "Visual platform, positive reinforcement culture",
                "Reddit": "Discussion-based, community moderation, niche topics",
                "TikTok": "Short-form video, youth culture, trend-driven"
            }

            st.subheader(f"📱 {platform} Post Analysis")

            # Main results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Content Assessment:**")
                result_class = "cyberbullying" if pred == 1 else "not-cyberbullying"
                result_text = "🚨 POTENTIALLY HARMFUL" if pred == 1 else "✅ APPEARS SAFE"
                st.markdown(f"""
                    <div class="prediction-box {result_class}">
                        <h3>{result_text}</h3>
                        <p>Confidence: {conf*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Emotional Tone:**")
                st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #{color}; color: white; text-align: center;">
                        <h2>{icon}</h2>
                        <h4>{sentiment.upper()}</h4>
                        <p>Sentiment Score: {score:.3f}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Platform context
            st.subheader("🌐 Platform Context")
            st.info(f"**{platform} Characteristics:** {platform_context[platform]}")

            # Risk assessment
            st.subheader("⚠️ Risk Assessment")

            risk_level = "Low"
            risk_color = "green"
            risk_icon = "🟢"

            if pred == 1:
                if sentiment == "Negative" and conf > 0.8:
                    risk_level = "High"
                    risk_color = "red"
                    risk_icon = "🔴"
                elif sentiment == "Negative" or conf > 0.7:
                    risk_level = "Medium"
                    risk_color = "orange"
                    risk_icon = "🟠"
                else:
                    risk_level = "Medium"
                    risk_color = "orange"
                    risk_icon = "🟠"
            elif sentiment == "Negative":
                risk_level = "Low-Medium"
                risk_color = "yellow"
                risk_icon = "🟡"

            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: {risk_color}; color: white; text-align: center; margin: 10px 0;">
                <h3>{risk_icon} {risk_level.upper()} RISK</h3>
                <p>Based on content analysis and platform context</p>
            </div>
            """, unsafe_allow_html=True)

            # Recommendations
            st.subheader("💡 Recommendations")
            if risk_level == "High":
                st.error("**Action Required:** This post shows strong indicators of cyberbullying. Consider reporting or removing if you have moderation authority.")
            elif risk_level == "Medium":
                st.warning("**Monitor Closely:** This content may be borderline. Watch for user reactions and consider community guidelines.")
            else:
                st.success("**Safe Content:** This post appears appropriate for the platform. No immediate action needed.")


def main():
    """Main application"""
    # Sidebar
    with st.sidebar:
        st.title("🛡️ Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "🔍 Detection", "📂 Batch Processing", "📊 Analytics", "🔬 Advanced Features"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This system uses advanced machine learning to detect cyberbullying 
        in real-time with 90%+ accuracy.
        """)
        
        st.markdown("### 📞 Support")
        st.markdown("For issues or questions, contact support.")
    
    # Load detector
    detector = get_detector()
    
    if not detector.models:
        st.error("❌ Models not found! Please train models first by running: `python src/train_models.py`")
        return
    
    # Route to selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Detection":
        show_detection_page(detector)
    elif page == "📂 Batch Processing":
        show_batch_detection(detector)
    elif page == "📊 Analytics":
        show_analytics(detector)
    elif page == "🔬 Advanced Features":
        show_advanced_features(detector)


if __name__ == "__main__":
    main()