import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_bar_chart(data, title=None):
    st.write(title if title else "Bar Chart")
    st.bar_chart(data)

def plot_heatmap(df_corr, title="Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def plot_wordcloud(text, width=800, height=400):
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=width, height=height, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
