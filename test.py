import instaloader
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Create an instance of Instaloader
L = instaloader.Instaloader()

# Login using your Instagram credentials (optional)
# L.login('your_username', 'your_password')

# Define the Instagram profile to scrape
profile_name = "zaeger_watches"

# Load the profile
profile = instaloader.Profile.from_username(L.context, profile_name)

# Collect data
posts = []
for post in profile.get_posts():
    posts.append({
        "Date": post.date,
        "Caption": post.caption,
        "Likes": post.likes,
        "Comments": post.comments,
    })

# Convert to DataFrame
df = pd.DataFrame(posts)

# Perform sentiment analysis on captions
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['Sentiment'] = df['Caption'].apply(lambda x: get_sentiment(x) if x else 0)

# Save the DataFrame to a CSV file
df.to_csv('instagram_engagment.csv', index=False)

# Visualize the data
sns.histplot(df['Sentiment'], bins=20, kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Engagement'] = df['Likes'] + df['Comments']
df['Engagement'].resample('D').mean().plot()
plt.title('Daily Average Engagement')
plt.xlabel('Date')
plt.ylabel('Engagement')
plt.show()
