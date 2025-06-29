
from urlextract import URLExtract
from wordcloud import WordCloud,STOPWORDS
from collections import Counter
import pandas as pd
import re
import emoji
extractor = URLExtract()
def fetch_stats(selected_user,df):
    if selected_user == 'Overall':

        num_messages = df.shape[0]
        words = []
        for message in df['message']:
            words.extend(message.split())

        num_media = df[df['message']=='<Media omitted>\n'].shape[0] 

        links = []
        for message in df['message']:
            links.extend(extractor.find_urls(message))
        return num_messages,words,num_media,links
    else:
        num_messages = df[df['user'] == selected_user].shape[0]
        words = []
        for message in df[df['user'] == selected_user]['message']:
            words.extend(message.split())
        num_media = df[df['message']=='<Media omitted>\n'].shape[0] 
        links = []
        for message in df[df['user'] == selected_user]['message']:
            links.extend(extractor.find_urls(message))
        return num_messages, words, num_media, links
    

def clean_message(msg):
    msg = re.sub(r'<Media omitted>', '', msg)
    msg = re.sub(r'\n', ' ', msg)
    msg = re.sub(r'http\S+|www\S+|https\S+', '', msg)  # Remove links
    msg = re.sub(r'\b[A-Za-z0-9]{8,}\b', '', msg)  # Remove long alphanumeric tokens
    return msg

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['message'] = df['message'].apply(clean_message)

    wc = WordCloud(width=500, height=500, min_font_size=10,
                   background_color='black', stopwords=STOPWORDS)

    df_wc = wc.generate(" ".join(df['message']))
    return df_wc

def most_common_words(selected_user, df, top_n=20):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['message'] = df['message'].apply(clean_message)

    words = []
    for message in df['message']:
        words.extend(message.split())

    most_common = Counter(words).most_common(top_n)
    return pd.DataFrame(most_common, columns=['word', 'count'])

def tot_emoji(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_counts = Counter(emojis)
    return pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count']).sort_values(by='count', ascending=False)


def timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline_df = df.groupby(['only_date']).count()['message'].reset_index()
    timeline_df.rename(columns={'message': 'message_count'}, inplace=True)
    return timeline_df



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import streamlit as st

# Load model once
@st.cache_resource
def load_toxic_model():
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
    return tokenizer, model

tokenizer, model = load_toxic_model()

# Toxicity Labels
LABELS = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

# Inference function
def detect_toxicity(messages, users, batch_size=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import pandas as pd

    # Load model & tokenizer once (you can move this outside if needed)
    model_name = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Dynamic whitelist from usernames (lowercase)
    whitelist = [u.lower() for u in users if isinstance(u, str)]

    def is_whitelisted(msg):
        return any(name in msg.lower() for name in whitelist)

    toxic_results = []
    filtered_msgs = [msg for msg in messages if isinstance(msg, str) and not is_whitelisted(msg)]

    LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_hate"]

    for i in range(0, len(filtered_msgs), batch_size):
        batch = filtered_msgs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).tolist()

        for text, score in zip(batch, scores):
            result = {label: round(s, 3) for label, s in zip(LABELS, score)}
            result['message'] = text
            toxic_results.append(result)

    return pd.DataFrame(toxic_results)

