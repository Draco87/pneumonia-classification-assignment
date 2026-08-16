import streamlit as st

from assistant.assistant import ask_assistant


st.set_page_config(
    page_title="Pneumonia ML Project Assistant",
    page_icon="🩻"
)


st.title("Pneumonia Classification Project Assistant")

st.write(
    "Ask questions about the problem, dataset, model, "
    "training procedure, evaluation results, limitations "
    "and potential improvements."
)

st.warning(
    "This is an experimental machine-learning project "
    "and is not intended for clinical diagnosis."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input(
    "Ask a question about the project..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)


    with st.chat_message("assistant"):

        with st.spinner("Generating response..."):

            try:
                answer = ask_assistant(question)

            except Exception as error:
                answer = f"Assistant error: {error}"

        st.write(answer)


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })