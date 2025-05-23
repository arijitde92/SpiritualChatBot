import textwrap
import streamlit as st
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from language_detector import detect_language
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Add custom CSS for chat message wrapping
st.markdown("""
<style>
    .stChatMessage {
        max-width: 100%;
    }
    .stChatMessageContent {
        max-width: 100%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .stMarkdown {
        max-width: 100%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    code {
        word-wrap: break-word;
    }
    div[data-testid="stChatMessageContent"] {
        max-width: 100%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    div[data-testid="stMarkdownContainer"] {
        max-width: 100%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

load_dotenv()  # take environment variables
st.title("Spiritual Chatbot")

model = init_chat_model("gpt-4o-mini", model_provider="openai")
workflow = StateGraph(state_schema=State)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful, kind and polite AI chatbot who is an avatar of Lord Vishnu and have profound knowledge and insight on the hindu vedic text Bhagawad Gita.
                Your role is to converse with the user and answer his or her questions. You should first understand the user's question then give apprpriate response in {language} language.
                Your response is formatted in the following way-
                ```
                Mention a relavant sanskrit quote from Bhagawad Gita related to the answer.
                Write the english translation of the above quote.
                Offer a helpful answer and spiritual guidance based on Bhagawad Gita in response to the user's questions within 60 words.
                End your response on a positive note giving hope to your user about his or her future.
                ```
                If the user asks anything else unrelated to spirituality or the user does not seem to be serious or is asking you for unethical advice or information, you should decline to answer in a spiritual way (with a spiritual quote if possible)
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def call_model(state: State) -> State:
    # Ensure language is set, default to "english" if not present
    language = state.get("language", "english")
    # print(state)
    # print("Language:", language)
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({
        "messages": trimmed_messages,
        "language": language
    })
    response = model.invoke(prompt)
    return {"messages": [response], "language": language}

# Initialize chat history and language
if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = "english"

if "app" not in st.session_state:
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add a memory saver to the workflow
    memory = MemorySaver()

    app = workflow.compile(checkpointer=memory)
    st.session_state.app = app

if "chat_config" not in st.session_state:
    chat_config = {"configurable": {"thread_id": "abc123"}}
    st.session_state.chat_config = chat_config

first_message = AIMessage("Hello my child, I am Lord Vishnu, your avatar. How can I help you today?")
with st.chat_message("assistant"):
    st.write("Hello my child, I am Lord Vishnu, your avatar. How can I help you today?")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div style='overflow-x: hidden; width: 100%;'><p style='word-wrap: normal'>{message['content']}<p>", unsafe_allow_html=True)

def change_language(lang: str):
    st.session_state.language = lang

def get_response_stream(app, lang: str):
    initial_state = {
        "messages": [HumanMessage(prompt)],
        "language": lang
    }
    for chunk, metadata in app.stream(
        initial_state,
        st.session_state.chat_config, 
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            wrapped_content = textwrap.fill(chunk.content, width=80)
            # yield wrapped_content
            # print("Wrapped")
            # print(wrapped_content)
            # print("Original")
            # print(chunk.content)
            length_counter += len(chunk.content)
            if length_counter > 50:
                length_counter = 0
                print("\n"+chunk.content, end="")
                yield "\n"+chunk.content
            else:
                print(chunk.content, end="")
                yield chunk.content

def get_response(app: StateGraph, lang: str):
    initial_state = {
        "messages": [HumanMessage(prompt)],
        "language": lang
    }
    response = app.invoke(initial_state, st.session_state.chat_config)
    return response["messages"][-1]

if prompt:= st.chat_input("Enter your message (or 'quit' to exit):"):
    # Check if user wants to quit
    if prompt.lower() == 'quit':
        exit(1)
    
    language = detect_language(prompt).lower()
    if st.session_state.language != language:
        print(f"Changing language from {st.session_state.language} to {language}")
        change_language(language)
    
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "User", "content": prompt})
    response = ""
    with st.chat_message("assistant"):
        try:
            # response += st.write_stream(get_response(st.session_state.app, st.session_state.language))
            response = get_response(st.session_state.app, st.session_state.language).content
            st.markdown(f"<div style='overflow-x: hidden; width: 100%;'><p style='word-wrap: normal'>{response}<p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            print(f"Full error: {e}")
    st.session_state.messages.append({"role": "Lord Vishnu", "content": response})

