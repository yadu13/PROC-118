import pandas as pd
import numpy as np
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

train_data = pd.read_csv("./static/assets/datafiles/updated_product_dataset.csv")
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model/sentiment_analysis_model.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)


# dictionary where key : emotion , value : list
encode_emotions = {
                    "Neutral": [0,"./static/assets/emoticons/neutral.png"],
                    "Positive": [1,"./static/assets/emoticons/positive.png"],
                    "Negative": [2,"./static/assets/emoticons/negative.png"]
                    }


def predict(text):

    sentiment = ""
    emoji_url = ""
    customer_review = []
    customer_review.append(text)

    sequences = tokenizer.texts_to_sequences(customer_review)

    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    result = model.predict(padded)
    label = np.argmax(result , axis=1)
    label = int(label)

    # extracting emotion and url from dictionary
    for emotion in encode_emotions:
        if encode_emotions[emotion][0]  ==  label:
            sentiment = emotion
            emoji_url = encode_emotions[emotion][1]

    return sentiment , emoji_url


#Display entry
def show_entry():
    day_entry_list = pd.read_csv("./static/assets/data_files/updated_product_dataset.csv")

    day_entry_list = day_entry_list.iloc[::-1]
    
    date1 = (day_entry_list['date'].values[0])
    date2 =(day_entry_list['date'].values[1])
    date3 = (day_entry_list['date'].values[2])

    entry1 = day_entry_list['text'].values[0]
    entry2 = day_entry_list['text'].values[1]
    entry3 = day_entry_list['text'].values[2]

    sentiment1 = day_entry_list["sentiment"].values[0]
    sentiment2 = day_entry_list["sentiment"].values[1]
    sentiment3 = day_entry_list["sentiment"].values[2]

    sentiment_url_1=""
    sentiment_url_2=""
    sentiment_url_3=""

    for key, value in encode_emotions.items():
        if key==sentiment1:
            sentiment_url_1 = value[1]
        if key==sentiment2:
            sentiment_url_2 = value[1]
        if key==sentiment3:
            sentiment_url_3 = value[1]

    return [
        {
            "date": date1,
            "entry": entry1,
            "sentiment": sentiment1,
            "sentiment_url": sentiment_url_1
        },
        {
            "date": date2,
            "entry": entry2,
            "sentiment": sentiment2,
            "sentiment_url": sentiment_url_2
        },
        {
            "date": date3,
            "entry": entry3,
            "sentiment": sentiment3,
            "sentiment_url": sentiment_url_3
        }
    ]
