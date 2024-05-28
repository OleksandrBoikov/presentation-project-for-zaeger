import instaloader
import pandas as pd

# Create an instance of Instaloader
L = instaloader.Instaloader()

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

# Preprocess the text data
df['Caption'] = df['Caption'].fillna('')
df['Caption'] = df['Caption'].apply(lambda x: x.lower())

# Save the DataFrame to a CSV file
df.to_csv('instagram_posts.csv', index=False)



