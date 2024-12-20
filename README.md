RAG 기반 LLM 애플리케이션 개발 (feat. LangChain)
프로젝트 요약
이 프로젝트는 LangChain과 Chroma를 활용하여 의료법 검색을 위한 Retrieval-Augmented Generation (RAG) 애플리케이션을 구현합니다. 사용자는 의료법에 대한 질문을 입력하면, 관련 데이터를 검색하여 정확하고 맥락에 맞는 답변을 제공합니다.

주요 특징:
의료법 지식베이스: 한국 의료법 문서를 데이터베이스로 구축하여 검색 및 질의 응답에 활용.
LangChain 기반 연결: LangChain의 RetrievalQA 체인을 활용하여 대형 언어 모델(LLM)과 연결.
향상된 검색 성능: 키워드 기반 검색과 유사도 검색을 결합하여 정확도를 높임.
LangChain과 Chroma를 활용한 RAG 구성
이 프로젝트는 LangChain과 Chroma를 기반으로 아래와 같은 워크플로우로 구성됩니다:

1. 데이터 준비 및 분할
문서 로드: Docx2txtLoader를 사용하여 한국 의료법 문서를 .docx 형식으로 불러옵니다.
문서 분할: RecursiveCharacterTextSplitter를 활용하여 문서를 작은 chunk로 분할하여 처리 효율성을 높입니다.
2. 데이터 임베딩 및 저장
임베딩 생성: UpstageEmbeddings의 "solar-pro" 모델을 사용하여 각 chunk를 벡터화합니다.
데이터 저장: Chroma를 사용하여 벡터화된 데이터를 영구적인 벡터 데이터베이스에 저장합니다.
3. 질문-답변 생성
유사도 검색: 사용자가 입력한 질문과 가장 관련성이 높은 chunk를 Chroma를 통해 검색합니다.
RetrievalQA 체인: LangChain의 RetrievalQA 체인을 활용하여 검색된 데이터를 기반으로 자연어 답변을 생성합니다.
현재 상태
의료법 검색 챗봇 애플리케이션인 medical_chatbot.py는 Streamlit 기반으로 구현되었습니다. 하지만, API KEY 문제로 인해 Streamlit 환경에서 실행 시 오류가 발생하고 있습니다. 동일한 API KEY로 Jupyter Notebook에서는 정상적으로 작동하며, 이 문제를 해결하기 위한 조사가 진행 중입니다.

프로젝트 목표
이 프로젝트는 의료법과 같은 특정 도메인에 특화된 정보를 검색하고, LLM을 활용해 자연어로 답변을 제공하는 솔루션을 구축하는 데 중점을 두고 있습니다. RAG 및 LangChain을 활용하여 효율적이고 확장 가능한 검색 시스템을 구현합니다.