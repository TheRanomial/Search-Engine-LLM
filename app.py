from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
import streamlit as st

wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wikipedia=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search here")


st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Please enter you Groq API key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if prompt:=st.chat_input(placeholder="Hey search anything"):
    
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    llm=ChatGroq(model="llama3-8b-8192",groq_api_key=api_key,streaming=True)
    tools=[arxiv,wikipedia,search]
    
    agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
    

    
    




