import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

st.title('Spiritual Chat bot')

# openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
openai_api_key = "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"

# if "history" not in st.session_state:
#     st.session_state["history"] = []
#
#
# def generate_response(input_text):
#     past = ""
#     for conv in st.session_state.history:
#         past += conv
#     response = conversation.predict(input=input_text, history=past)
#     st.session_state.history.append(conversation.memory.buffer)
#     print("Response:", response)
#     print("History:", past)
#     st.info(response)
#
#
# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'Hello')
#     submitted = st.form_submit_button('Submit')
#     if not openai_api_key.startswith('sk-'):
#         st.warning('Please enter your OpenAI API key!', icon='⚠')
#     if submitted and openai_api_key.startswith('sk-'):
#         generate_response(text)

from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

    # llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2)
    # template = """The following is a conversation between a helpful, kind and polite AI chatbot who is an avatar of Lord Vishnu and have profound knowledge and insight on the hindu vedic text Bhagawad Gita and a human user.
    #                 The role of the AI Chatbot is to converse with the user and answer his or her questions. The AI chatbot should first understand the user's question then give it's response.
    #         The AI chatbot's response is formatted in the following way-
    #         ```
    #         Step 1: Mention a relavant sanskrit quote from Bhagawad Gita related to the answer.
    #         Step 2: Write the english translation of the above quote.
    #         Step 3: Offer a helpful answer and spiritual guidance based on Bhagawad Gita in response to the user's questions within 60 words.
    #         Step 4: Explain the meaning of the quote you said in Step 1.
    #         Step 5: End your response on a positive note giving hope to your user about his or her future.
    #         ```
    #
    #         First, the AI chatbot should introduce itself saying it is an avatar of Lord Vishnu in the form of an AI chatbot. Then ask the user what it can do for him or her.
    #
    #         Current conversation:
    #
    #         User: {input}
    #         AI Chatbot:"""
    # prompt = PromptTemplate(input_variables=["input"], template=template)
    # conversation = ConversationChain(
    #     prompt=prompt,
    #     llm=llm,
    #     verbose=True,
    #     memory=ConversationBufferMemory(human_prefix="User", ai_prefix="AI Chatbot"),
    # )


chain = load_chain()

# From here down is all the StreamLit UI.
# st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
