import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st
from docx import Document
from pypdf import PdfReader

from rag import ask, index_documents


def extract_text(uploaded_file) -> str:
    """Read text from a supported uploaded file."""
    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()
    data = uploaded_file.getvalue()

    if extension == ".txt":
        return data.decode("utf-8-sig")

    if extension == ".pdf":
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if extension == ".docx":
        document = Document(BytesIO(data))
        return "\n".join(
            paragraph.text for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

    raise ValueError(f"Unsupported file type: {extension}")


st.set_page_config(page_title="Ask My Documents")
st.title("Ask My Documents")
st.caption("Attach .txt, .pdf, or .docx files in the chat, then ask questions.")

# These values belong to this Streamlit browser session.
if "records" not in st.session_state:
    st.session_state.records = []

if "seen_files" not in st.session_state:
    st.session_state.seen_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear documents and chat"):
    st.session_state.records = []
    st.session_state.seen_files = set()
    st.session_state.messages = []
    st.rerun()

st.write(
    f"Searchable passages in this session: "
    f"{len(st.session_state.records)}"
)

# Redraw previous chat messages after Streamlit reruns the script.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

submission = st.chat_input(
    "Attach documents and ask a question",
    accept_file="multiple",
    file_type=["txt", "pdf", "docx"],
    max_upload_size=10,  # MB per file
)

if submission:
    question = submission.text.strip()
    uploaded_files = submission.files

    if uploaded_files:
        documents_to_index = []
        new_fingerprints = []

        for uploaded_file in uploaded_files:
            data = uploaded_file.getvalue()
            fingerprint = hashlib.sha256(data).hexdigest()

            # Avoid indexing the exact same file twice in this session.
            if fingerprint in st.session_state.seen_files:
                continue

            try:
                extracted = extract_text(uploaded_file).strip()

                if not extracted:
                    st.warning(
                        f"No selectable text found in "
                        f"{uploaded_file.name}."
                    )
                    continue

                documents_to_index.append(
                    (uploaded_file.name, extracted)
                )
                new_fingerprints.append(fingerprint)

            except Exception as error:
                st.error(
                    f"Could not read {uploaded_file.name}: {error}"
                )

        if documents_to_index:
            try:
                with st.spinner("Indexing uploaded documents..."):
                    new_records = index_documents(documents_to_index)

                st.session_state.records.extend(new_records)
                st.session_state.seen_files.update(new_fingerprints)

                names = ", ".join(
                    name for name, _ in documents_to_index
                )
                st.success(f"Added: {names}")

            except Exception as error:
                st.error(f"Could not index documents: {error}")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            if not st.session_state.records:
                answer = "Attach a document first, then ask a question."
                st.write(answer)
            else:
                try:
                    with st.spinner("Searching your documents..."):
                        answer, passages = ask(
                            question,
                            st.session_state.records,
                        )

                    st.write(answer)

                    with st.expander("See retrieved passages"):
                        for number, passage in enumerate(
                            passages, start=1
                        ):
                            st.markdown(
                                f"**[{number}] "
                                f"{passage['source']} — "
                                f"chunk {passage['chunk_number']}**"
                            )
                            st.write(passage["text"])

                except Exception as error:
                    answer = f"Could not answer: {error}"
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })