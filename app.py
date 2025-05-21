import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Custom CSS for chat message wrapping
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
</style>
""", unsafe_allow_html=True)

load_dotenv()  # take environment variables
st.title("Spiritual Chatbot")

model = init_chat_model("gpt-4o-mini", model_provider="openai")
workflow = StateGraph(state_schema=MessagesState)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful, kind and polite AI chatbot who is an avatar of Lord Vishnu and have profound knowledge and insight on the hindu vedic text Bhagawad Gita.
                Your role is to converse with the user and answer his or her questions. You should first understand the user's question then give it's response.
                Your response is formatted in the following way-
                ```
                Mention a relavant sanskrit quote from Bhagawad Gita related to the answer.
                Write the english translation of the above quote.
                Offer a helpful answer and spiritual guidance based on Bhagawad Gita in response to the user's questions within 60 words.
                End your response on a positive note giving hope to your user about his or her future.
                ```
            """,
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

def call_model(state: MessagesState) -> MessagesState:
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add a memory saver to the workflow
memory = MemorySaver()

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("What is on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    # Get bot response
    input_msgs = [HumanMessage(prompt)]
    with st.chat_message("assistant"):
        try:
            response = ""
            for chunk, metadata in app.stream({"messages": input_msgs}, config, stream_mode="messages"):
                if isinstance(chunk, AIMessage):
                    response += chunk.content
                    st.markdown(response, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

