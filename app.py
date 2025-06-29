import streamlit as st
import preprocessor
import helper
from matplotlib import pyplot as plt
st.sidebar.title("Analysing Whatsapp Chat Data by Akshit")
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    st.text(data[:1000])  # Display first 1000 characters of the file
    df = preprocessor.preprocess(data)
    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')  # Remove group notifications if present
    
    user_list.sort()
    user_list.insert(0, "Overall")  # Optional: Add overall option
    

    selected_user = st.sidebar.selectbox("Show Analysis with respect to", user_list)
    if st.sidebar.button("Show Analysis"):
       
       num_messages,words,num_media,links =helper.fetch_stats(selected_user,df)
       
       col1,col2,col3,col4 =  st.columns(4)

       with col1:
          st.header("Total Chats of "+ selected_user)
          st.title(num_messages)

       with col2:
          st.header("Total Words of "+ selected_user)
          st.title(len(words))

       with col3:
          st.header("Total Media Shared by "+ selected_user)
          st.title(num_media)  

       with col4:
          st.header("Links Shared by "+ selected_user)
          st.title(len(links))

       if selected_user == 'Overall':
           col1,col2 = st.columns(2)
           with col1:
                st.header("Most Active Users")
                x = df['user'].value_counts().head()
                st.bar_chart(x) 

           with col2:
              st.header("Most Active Users Percentage")

              user_percent = (
                   df['user'].value_counts(normalize=True) * 100
               ).round(2).reset_index().rename(columns={'index': 'name', 'user': 'percentage'})

              st.dataframe(user_percent)

    df_wc = helper.create_wordcloud(selected_user, df)
    st.subheader("Word Cloud")
    st.image(df_wc.to_array())
    
    st.subheader("Most Common Words")
    most_common_df = helper.most_common_words(selected_user, df)
    st.dataframe(most_common_df)
    #show bar graph of most common words
    st.bar_chart(most_common_df.set_index('word')['count'])

    emoji_List = helper.tot_emoji(selected_user, df)
    st.subheader("Most Common Emojis")
    st.dataframe(emoji_List)
    st.bar_chart(emoji_List.set_index('emoji')['count'].head(10))

    st.subheader("Timeline")
    timeline_df = helper.timeline(selected_user, df)
    st.scatter_chart(timeline_df, x='only_date', y='message_count')

    if st.sidebar.checkbox("Detect Toxic Messages"):
        with st.spinner("Detecting toxicity..."):
           all_users = df['user'].unique().tolist()
           toxic_df = helper.detect_toxicity(df['message'].tolist(), all_users)

        st.subheader("Toxic Messages Detected")
        threshold = 0.7
        toxic_filtered = toxic_df[toxic_df['toxicity'] > threshold]

        if toxic_filtered.empty:
           st.success("No toxic messages detected above threshold.")
        else:
           st.dataframe(toxic_filtered[['message', 'toxicity', 'insult', 'obscene']])

        # Pie chart showing distribution
        st.subheader("Toxic vs Non-Toxic Distribution")
        toxic_count = len(toxic_filtered)
        clean_count = len(toxic_df) - toxic_count

        fig, ax = plt.subplots(facecolor='black')
        wedges, texts, autotexts = ax.pie(
            [toxic_count, clean_count],
            labels=["Toxic", "Clean"],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': "white"},
            colors=['#ff4d4d', '#33cc33']  # Red for toxic, green for clean
        )
        ax.set_facecolor("black")
        ax.set_title("Toxicity Distribution", color='white')

        st.pyplot(fig)
 # ✅ Pass the figure, not plt.pie directly
else:
    st.sidebar.info("Please upload a WhatsApp chat file (.txt) to continue.")
