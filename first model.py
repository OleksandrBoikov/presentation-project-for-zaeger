import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
from textblob import TextBlob


df=pd.read_csv('instagram_posts.csv')


# Perform sentiment analysis on captions
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['Sentiment'] = df['Caption'].apply(lambda x: get_sentiment(x) if x else 0)

print(df.head())  # Check the DataFrame with sentiment scores

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Caption'])
sequences = tokenizer.texts_to_sequences(df['Caption'])
padded_sequences = pad_sequences(sequences, maxlen=100)

# Verify the existence of the 'Sentiment' column
if 'Sentiment' not in df.columns:
    raise KeyError("The 'Sentiment' column is missing from the DataFrame.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['Sentiment'], test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict sentiment for new data
predictions = model.predict(padded_sequences)
df['Predicted_Sentiment'] = predictions

# Save the DataFrame with predictions
df.to_csv('instagram_posts_with_predictions.csv', index=False)

# Visualize the sentiment distribution
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Predicted_Sentiment'], bins=20, kde=True)
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
