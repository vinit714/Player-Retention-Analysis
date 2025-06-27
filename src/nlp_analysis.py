import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

nltk.download('stopwords')

def load_reviews(filepath: str) -> pd.DataFrame:
    """
    Load player reviews dataset.
    Args:
        filepath (str): Path to CSV containing reviews.
    Returns:
        pd.DataFrame with 'review' column.
    """
    df = pd.read_csv(filepath)
    return df

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercasing, removing stopwords, stemming.
    Args:
        text (str): Raw text.
    Returns:
        str: Cleaned text.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered_tokens)

def add_cleaned_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a cleaned review text column.
    """
    df['cleaned_review'] = df['review'].apply(clean_text)
    return df

def generate_wordcloud(text: str):
    """
    Generate a WordCloud plot for the given text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment polarity using TextBlob.
    Adds a 'sentiment' column: positive, neutral, or negative.
    """
    def polarity_label(p):
        if p > 0.1:
            return 'positive'
        elif p < -0.1:
            return 'negative'
        else:
            return 'neutral'
    df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment'] = df['polarity'].apply(polarity_label)
    return df

def topic_modeling(df: pd.DataFrame, n_topics=5):
    """
    Apply LDA topic modeling on cleaned reviews.
    Display top words per topic.
    """
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['cleaned_review'])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    st.subheader(f"Top words per topic (LDA, {n_topics} topics)")
    for i, topic in enumerate(lda.components_):
        st.write(f"Topic {i+1}: ", [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:][::-1]])

def run_nlp_dashboard(review_csv_path: str):
    """
    Streamlit dashboard to run NLP analysis.
    """
    st.title("🎙️ Player Review NLP Analysis")
    df = load_reviews(review_csv_path)
    df = add_cleaned_reviews(df)
    df = sentiment_analysis(df)

    st.write("Sample reviews:")
    st.write(df[['review', 'sentiment']].head())

    all_text = " ".join(df['cleaned_review'].tolist())
    st.subheader("WordCloud of Reviews")
    generate_wordcloud(all_text)

    topic_modeling(df, n_topics=5)

    st.subheader("Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())
