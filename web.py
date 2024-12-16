import streamlit as st
import predict

st.set_page_config(page_title="Text Summarizer", page_icon="📝")

st.title("📝 Text Summarizer")

# Input text area
text_input = st.text_area("Enter the text you want to summarize:", height=250)

# Optional settings for the summarization
col1, col2 = st.columns(2)

with col1:
    extraction_method = st.selectbox(
        "Model", options=["Pointer Generator Coverage Mechanism"]  # Replace with your backend methods
    )

with col2:
    st.write("")  # Empty column for spacing
    st.write("")
    summarize_button = st.button("Summarize")

# Summarization process
if summarize_button:
    if text_input:
        with st.spinner("Summarizing..."):
            pred = predict.Predict()
            summary = pred.predict(text_input.split(), beam_search=True)
            st.write("**Summary:**")
            st.success(summary)  # Display the summary
    else:
        st.warning("Please enter some text to summarize.")

# Footer with additional information (smaller text)
st.markdown(
    """
    <hr>
    <div style="font-size: 12px; text-align: center; color: gray;">
        <p>Thank you for using our text summarization app! We hope it saved you time and made things simpler. If you have any feedback or ideas, we’d love to hear them. Stay tuned for updates—there’s more to come!</p>
        <p>
            <strong>Connect with us:</strong><br>
            📧 Email: <a href="mailto:nkluong2003@gmail.com" style="color: gray;">nkluong2003@gmail.com</a><br>
            🌐 Website: <a href="https://www.textsummarizer.com" target="_blank" style="color: gray;">Text Summarizer</a><br>
            📘 Facebook: <a href="https://www.facebook.com/iam.lkng/" target="_blank" style="color: gray;">Follow us on Facebook</a><br>
            🐦 Twitter: <a href="https://twitter.com/TextSummarizer" target="_blank" style="color: gray;">Follow us on Twitter</a><br>
            Made by LKN & MingYong
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
