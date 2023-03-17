#importing the libraries 
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ConversationChain
import pywebio
from pywebio.input import input,select
from pywebio.output import put_text, put_html, put_markdown 
import os
from langchain.document_loaders import PyPDFLoader
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import urllib.request as urllib2

def questions(loader):
    api_Key='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'
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    embeddings = OpenAIEmbeddings(openai_api_key=api_Key)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=api_Key), chain_type="stuff", vectorstore=db)
    count=0
    while count>=0:
        x=input("Customer Response")
        if x!="":
            put_markdown("**Customer:** \n"+x).style('text-align: left')
            ans=qa.run(x)
            put_markdown("**AmplifAI LIEA:** \n"+ans).style('text-align: right')  

def main():
    api_Key='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'
    
    put_markdown('## AmplifAI Training Simulator').style('text-align: center')
    #api_Key=os.getenv("api_Key")
    put_markdown("**Agent:** \n"+"Hello, Thank you for reaching out to Samsung customer care. Please select the item you want help with from the list below.").style('text-align: right')    
    y=select("Intent", options=["",'Smart TV','Smart Fridge'])
    if y=="Smart TV":
        put_markdown("**Customer:** \n"+"Product selected: Smart TV").style('text-align: left')
        put_markdown("**Agent:** \n"+"Can you please tell me model number of your Smart TV ?").style('text-align: right')
        put_markdown("**Customer:** \n"+"Model number is STV-OLEDRF7000").style('text-align: left')
        put_markdown("**Agent:** \n"+"Thank you for providing the model number. Can you please tell me more about your issue with the Smart TV ?").style('text-align: right')
        loader = PyPDFLoader('D:\\Trouble Shooting\\smart_tv.pdf')
        questions(loader)

    elif y=="Smart Fridge":
        put_markdown("**Customer:** \n"+"Product selected: Smart Fridge").style('text-align: left')
        put_markdown("**Agent:** \n"+"Can you please tell me model number of your Smart Fridge ?").style('text-align: right')
        put_markdown("**Customer:** \n"+"Model number is SFridge-OLEDRF7000").style('text-align: left')
        put_markdown("**Agent:** \n"+"Thank you for providing the model number. Can you please tell me more about your issue with the Smart Fridge ?").style('text-align: right')
        loader = PyPDFLoader('D:\\Trouble Shooting\\samsung_fridge.pdf')
        questions(loader)

if __name__ == "__main__":
    import argparse
    from pywebio.platform.tornado_http import start_server as start_http_server
    from pywebio import start_server as start_ws_server

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("--http", action="store_true", default=False, help='Whether to enable http protocol for communicates')
    args = parser.parse_args()

    if args.http:
        start_http_server(main, port=args.port)
    else:
        # Since some cloud server may close idle connections (such as heroku),
        # use `websocket_ping_interval` to  keep the connection alive
        start_ws_server(main, port=args.port, websocket_ping_interval=30)                 

    