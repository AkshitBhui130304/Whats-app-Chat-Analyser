﻿# Whats-app-Chat-Analyser
# 📊 WhatsApp Chat Analyser with Toxicity Detection

🚀 **Live App**: [Click here to try it](https://whats-app-chat-analyser-8miprur8cgrg6mnvz5uxit.streamlit.app/)

This project is a Streamlit-based web application that analyzes **WhatsApp chat exports** and identifies **toxic or abusive messages** using AI models.

---

## ✅ Features

- 📈 **Chat Statistics** (per user or overall):
  - Total messages
  - Word count
  - Media shared
  - Links sent

- 🧠 **Toxic Message Detection** using pre-trained NLP models:
  - Toxicity
  - Insults
  - Obscene content

- 📆 **Timeline** of chat activity
- 🔥 **Word Cloud** for message visualization
- 😂 **Most Used Emojis**
- 💬 **Most Common Words**
- 👥 **Most Active Users**

---

## 📂 How to Use

1. **Export WhatsApp Chat (Without Media)**  
   - Open any chat on WhatsApp  
   - Tap `⋮` (three dots) → More → Export chat → Choose **"Without Media"**  
   - Save the `.txt` file

2. **Open the Web App**  
   [whats-app-chat-analyser.streamlit.app](https://whats-app-chat-analyser-8miprur8cgrg6mnvz5uxit.streamlit.app/)

3. **Upload the `.txt` File**  
   - Select a user or view overall analysis
   - Optionally check "Detect Toxic Messages" for abuse detection

---

## 🔐 Notes

- Toxicity detection uses `transformers` library and multilingual models.
- Messages with names from a whitelist (e.g., your own name) can be excluded from detection.
- The app works **completely client-side**—your chat is never stored or shared.

---

## 🧠 Technologies Used

- Python
- Streamlit
- Matplotlib / Seaborn
- Hugging Face Transformers (for toxicity classification)
- WordCloud
- Pandas

---

Feel free to ⭐️ the repository and contribute!
