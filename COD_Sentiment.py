import praw
import pandas as pd
from datetime import datetime


reddit = praw.Reddit(
    client_id="kTnTOAPJ1PyoB3U_akcLPQ",
    client_secret="dR0tQdFEEiGjakdUCMVs3l-8MAvMTQ",
    user_agent="callofduty_analysis"
)

# 📥 Search posts in top COD subreddits
subreddits = ["modernwarfare", "CallOfDuty", "BlackOps6", "BlackOps", "COD", "ModernWarfareIII", "Warzone"]
search_terms = ["event pass", "battle pass"]
posts = []


# Season date ranges
season_ranges = {
    "Season 1": (datetime(2024, 11, 14), datetime(2025, 1, 27)),
    "Season 2": (datetime(2025, 1, 28), datetime(2025, 3, 31)),
    "Season 3": (datetime(2025, 4, 3), datetime(2025, 5, 28)),
    "Season 4": (datetime(2025, 5, 29), datetime(2025, 7, 10))
}



for sub in subreddits:
    for term in search_terms:
        for post in reddit.subreddit(sub).search(term, time_filter="all", limit=1000):
            created_dt = datetime.utcfromtimestamp(post.created_utc)
            
            for season, (start, end) in season_ranges.items():
                if start <= created_dt <= end:
                    posts.append({
                        "Subreddit": sub,
                        "Query": term,
                        "Season": season,
                        "Date": created_dt,
                        "Title": post.title,
                        "Body": post.selftext,
                        "Text": post.title + " " + post.selftext,
                        "Upvotes": post.score,
                        "Comments": post.num_comments
                    })
                    break  # Stop after finding the matching season

df = pd.DataFrame(posts)

import re

# text = "cod bring back loot boxes i am so sick of the boring battle passes nowadays no risk no flair just bank details and your item im not saying like cod remastered where you can only unlock guns via lootboxes but id love to see smaller content unlocked via free lootboxes the battlepasses are the most boring system going and as someone who plays cod quite frequently there arent nearly enough events loot boxes would be good to keep the game fresh rainbow six siege has done this and made a whole economy and marketplace because of it"

# from textblob import TextBlob
# print(TextBlob(text).sentiment.polarity)

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk

# # Download lexicon (only once)
# nltk.download('vader_lexicon')

# # Initialize VADER
# sia = SentimentIntensityAnalyzer()

# # Get sentiment scores
# scores = sia.polarity_scores(text)

# # Output
# print("VADER Sentiment Scores:")
# print(scores)




# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"[^a-z\s]", "", text) # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

# Apply cleaning
df["CleanText"] = df["Text"].apply(clean_text)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load pretrained model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_roberta_sentiment_full(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**tokens).logits
    scores = softmax(logits.numpy()[0])
    labels = ['Negative', 'Neutral', 'Positive']
    
    top_label = labels[scores.argmax()]
    return pd.Series({
        "RobertaSentiment": top_label,
        "Roberta_Positive": scores[2],
        "Roberta_Neutral": scores[1],
        "Roberta_Negative": scores[0]
    })


df_sentiment = df["CleanText"].apply(get_roberta_sentiment_full)
df = pd.concat([df, df_sentiment], axis=1)

# from textblob import TextBlob

# df["PolarityScore"] = df["Text"].apply(lambda x: TextBlob(x).sentiment.polarity)

# def get_sentiment(text):
#     score = TextBlob(text).sentiment.polarity
#     if score > 0.1:
#         return "Positive"
#     elif score < -0.1:
#         return "Negative"
#     else:
#         return "Neutral"

# df["Sentiment"] = df["Text"].apply(get_sentiment)


import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment by Season and Query
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Season", hue="RobertaSentiment", palette="pastel", order=["Season 1", "Season 2", "Season 3", "Season 4"])
plt.title("Overall Sentiment per Season")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Season", hue="Query", palette="muted", order=["Season 1", "Season 2", "Season 3", "Season 4"])
plt.title("Event Pass vs Battle Pass Mentions per Season")
plt.show()

# Stacked bar plot
colors = {
    'Positive': '#2ca02c',  # green
    'Neutral': '#1f77b4',   # blue
    'Negative': '#d62728'   # red
}

sentiment = df.groupby(["Query", "Season", "RobertaSentiment"]).size().unstack().fillna(0)
sentiment = sentiment[["Positive", "Neutral", "Negative"]]  # Order columns

sentiment.plot(kind="bar", stacked=True, figsize=(12, 6), color=[colors[col] for col in sentiment.columns])
plt.title("RobertaSentiment per Season per Pass")
plt.ylabel("Post Count")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()



from wordcloud import WordCloud
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')

custom_stopwords = set(stopwords.words('english') + list(string.punctuation))
custom_stopwords.update([
    'call', 'duty', 'event', 'pass', 'battle', 'game', 'season', 'new', 'black', 'ops', 'bo6'
])

def generate_clean_wordcloud(df_sub, title):
    words = ' '.join(df_sub['Text']).lower().split()
    words = [word for word in words if word.isalpha() and word not in custom_stopwords]
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

generate_clean_wordcloud(df[df["Query"] == "event pass"], "Event Pass – Refined Word Cloud")
generate_clean_wordcloud(df[df["Query"] == "battle pass"], "Battle Pass – Refined Word Cloud")


import seaborn as sns

# Filter only positive and negative sentiments
filtered_df = df[df["RobertaSentiment"].isin(["Positive", "Negative"])]

# Count per season and query
trend_data = filtered_df.groupby(["Season", "Query", "RobertaSentiment"]).size().reset_index(name="Count")

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_data, x="Season", y="Count", hue="Query", style="RobertaSentiment", markers=True, dashes=False)
plt.title("Positive vs Negative Sentiment Trend by Season")
plt.ylabel("Post Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

from collections import Counter

def get_top_words(df_sub, top_n=20):
    words = ' '.join(df_sub['Text']).lower().split()
    words = [word for word in words if word.isalpha() and word not in custom_stopwords]
    return Counter(words).most_common(top_n)

print("🔹 Top words for Event Pass:")
print(get_top_words(df[df["Query"] == "event pass"]))

print("\n🔹 Top words for Battle Pass:")
print(get_top_words(df[df["Query"] == "battle pass"]))


# 🔹 Top words for Event Pass:
# [('like', 130), ('get', 110), ('play', 77), ('really', 72), ('cod', 71), ('xp', 67), ('would', 61), ('think', 59), ('one', 53), ('double', 53), ('know', 51), ('even', 48), ('anyone', 47), ('still', 46), ('people', 45), ('playing', 44), ('also', 43), ('time', 41), ('getting', 40), ('weapon', 39)]

# 🔹 Top words for Battle Pass:
# [('get', 92), ('like', 91), ('cod', 87), ('play', 59), ('xp', 59), ('buy', 56), ('time', 56), ('anyone', 56), ('even', 47), ('know', 45), ('would', 44), ('one', 43), ('every', 43), ('back', 43), ('way', 42), ('still', 42), ('want', 41), ('got', 39), ('complete', 37), ('tokens', 37)]


#average polarity score by Season and passimport seaborn as sns

# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x="Season", y="PolarityScore", hue="Query", ci=None)
# plt.title("Average Sentiment Score by Season and Pass Type")
# plt.axhline(0, color='gray', linestyle='--')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

df["RobertaPolarity"] = df["Roberta_Positive"] - df["Roberta_Negative"]

season_order = ["Season 1", "Season 2", "Season 3", "Season 4"]
df["Season"] = pd.Categorical(df["Season"], categories=season_order, ordered=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Season", y="Roberta_Positive", hue="Query", ci=None)
plt.title("Average Roberta positive Sentiment Score by Season and Pass Type")
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Season", y="Roberta_Negative", hue="Query", ci=None)
plt.title("Average Roberta negetive Sentiment Score by Season and Pass Type")
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#  Top 10 Most Upvoted Posts and Sentiment Breakdown
top_upvoted = df.sort_values(by="Upvotes", ascending=False).head(10)
print("🔝 Top 10 Upvoted Sentiments:")
print(top_upvoted["RobertaSentiment"].value_counts())

# Top 10 Most Commented Posts and Sentiment Breakdown
top_commented = df.sort_values(by="Comments", ascending=False).head(10)
print("💬 Top 10 Most Commented Sentiments:")
print(top_commented["RobertaSentiment"].value_counts())


# 🔝 Top 10 Upvoted Sentiments:
# RobertaSentiment
# Negative    4
# Neutral     4
# Positive    2
# Name: count, dtype: int64


# 💬 Top 10 Most Commented Sentiments:
# RobertaSentiment
# Negative    7
# Positive    2
# Neutral     1
# Name: count, dtype: int64


import re
from collections import Counter

def extract_context_phrases(df, phrase="battle pass", window=4):
    context_phrases = []

    for text in df["CleanText"]:
        tokens = text.lower().split()
        for i in range(len(tokens) - 1):
            if tokens[i] == phrase.split()[0] and tokens[i+1] == phrase.split()[1]:
                start = max(i - window, 0)
                end = min(i + 2 + window, len(tokens))
                context = tokens[start:end]
                context_phrase = ' '.join(context)
                context_phrases.append(context_phrase)
    
    return Counter(context_phrases).most_common(30)

battle_contexts = extract_context_phrases(df[df["Query"] == "battle pass"])
event_contexts = extract_context_phrases(df[df["Query"] == "event pass"])

print("🔹 Battle Pass Contexts:")
for phrase, count in battle_contexts:
    print(f"{phrase} — {count}")

print("\n🔸 Event Pass Contexts:")
for phrase, count in event_contexts:
    print(f"{phrase} — {count}")


df.to_csv(r'CallOfDutyRSenti.csv', index=False) 
