import streamlit as st

from rag import ask

st.set_page_config(page_title="Ask My Notes")
st.title("Ask My Notes")
st.caption("A small RAG project: retrieve passages, then answer from them.")

question = st.chat_input("Ask a question about your documents")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes..."):
                answer, passages = ask(question)

            st.write(answer)

            with st.expander("See retrieved passages"):
                for number, passage in enumerate(passages, start=1):
                    st.markdown(
                        f"**[{number}] {passage['source']} — "
                        f"chunk {passage['chunk_number']}**"
                    )
                    st.write(passage["text"])

        except Exception as error:
            st.error(f"Could not answer: {error}")