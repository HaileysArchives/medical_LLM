import streamlit as st
import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv # 환경변수 로드 
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_upstage import ChatUpstage

st.set_page_config(page_title="의료법 챗봇", page_icon="💻")

st.title("💻의료법 챗봇")
st.caption("의료법에 관련된 모든 것을 답해드립니다.")

load_dotenv()

api_key = os.getenv("UPSTAGE_API_KEY")

# 데이터 로드 및 문서 분리 
loader = Docx2txtLoader('./medical.docx')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
document_list = loader.load_and_split(text_splitter=text_splitter)

# Embedding 생성
embedding = UpstageEmbeddings(model="embedding_query")

# # Chroma 데이터베이스 초기화 
# database = Chroma.from_documents(
#     document= document_list,
#     embedding=embedding,
#     collection_name='medical_law',
#     persist_directory='./chroma'
# )

persist_directory = "./chroma"

# 데이터베이스 로드 또는 생성
if os.path.exists(persist_directory):
    # 기존 데이터베이스 로드
    database = Chroma(
        collection_name='medical-law',
        persist_directory=persist_directory,
        embedding_function=embedding
    )
else:
    # 새로운 데이터베이스 생성
    database = Chroma.from_documents(
        documents=document_list,
        embedding=embedding,
        collection_name='medical-law',
        persist_directory=persist_directory
    )

# LLM 및 프롬프트 설정
llm = ChatUpstage(model="solar-pro", api_key=api_key)
prompt = hub.pull("rlm/rag-prompt")

# RetrievalQA 체인
qa_chain = RetrievalQA.from_chain_type(
    llm, 
    retriever = database.as_retriever(),
    chain_type_kwargs = {"prompt": prompt}
)

# AI 답변 생성 함수
def get_ai_message(user_message):
    ai_response = qa_chain.invoke({"query": user_message})
    return ai_response["result"]

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지 표시 
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 처리
if user_question := st.chat_input(placeholder="의료법에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # AI 응답 생성 및 표시 
    with st.chat_message("ai"):
        ai_message = get_ai_message(user_question)
        st.write(ai_message)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})





