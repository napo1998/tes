import streamlit as st

from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS




import tempfile
import os
from PIL import Image


# Loading Image using PIL
im = Image.open('./image/chatbot.png')
# Adding Image to web app
st.set_page_config(page_title="TRIPAI", page_icon = im)
# bg = Image.open('./image/background.png')
# st.image(bg, caption='None',use_column_width=True)
os.environ['OPENAI_API_KEY'] = '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'

with st.sidebar:
        st.title("Hi,I'm your customized CSVChatbot 🤖,please insert the csv file you want to analyze as below")

uploaded_file = st.sidebar.file_uploader("upload", type="csv")


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             
               background: #C9D6FF;  
               background: -webkit-linear-gradient(to right, #E2E2E2, #C9D6FF);  
               background: linear-gradient(to right, #E2E2E2, #C9D6FF);          
            
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()


#LangChain CSVLoader class allows us to split a CSV file into unique rows
if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())        
        tmp_file_path = tmp_file.name
                 

        
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()     
    

    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

   


    
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.1, verbose=True),retriever=vectors.as_retriever())
    

    def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Nice to meet you ! Ask me anything about " + uploaded_file.name + " 📜"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")