from langchain import PromptTemplate
from rag.vector_store import init_vectorstore
from embeddings.jina import JinaEmbedding as Jina
import os
from models.gemini import Gemini

# INIT MODELS
embedding_model = Jina()

llm_prompt_template = PromptTemplate(
    template="""
Hãy trở thành một trợ lý ảo chuyên nghiệp của trường Đại học Sư phạm Kỹ thuật TP.HCM. Bạn sẽ trả lời các câu hỏi của sinh viên về trường học chính xác nhất. Là người đại diện của trường bạn sẽ xưng hô là mình và bạn trong mọi trường hợp, bạn cần trả lời chính xác từ ác thông tin được cung cấp.
        Hãy đóng vai trò của mình và trả lời câu hỏi sau:
        Câu hỏi: {question}
        Ngữ cảnh: {context}
        Trả lời:

    """,
    input_variables=["question", "context"]
)

step_back_llm_prompt_template = PromptTemplate(
    template="""
    Bạn là một trợ lý AI tư vấn tuyển sinh của Trường Đại Học Sư Phạm Kỹ Thuật TP. Hồ Chí Minh (SPKT hoặc HCMUTE).   
    Nhiệm vụ của bạn là viết lại câu hỏi của người dùng sao cho rõ ràng, đầy đủ và dễ hiểu hơn,  
    nhưng vẫn giữ nguyên ý nghĩa gốc.  Nếu câu hỏi là về tư vấn ngành hãy làm rõ câu hỏi đó ra để giúp bạn hiểu hơn về ngành.

    Hãy đảm bảo câu hỏi sau khi chỉnh sửa có đủ ngữ cảnh và không gây hiểu lầm.  Hãy từ chối việc trả lời các câu hỏi vi phạm pháp luật hoặc không phù hợp với đạo đức.
    1. Hiện tại trường không còn ngành đào tạo chất lượng cao

    Câu hỏi cần chỉnh sửa: "{question}"  
    Câu hỏi đã chỉnh sửa:
    """,
    input_variables=["question"]
)


vec_db = init_vectorstore(data_path="./data/V2_DATASET.csv", db_name="hcmute-admission-sample", embed_columns=["title", "content"], embedding_model=embedding_model,rerank_top_k=10)
agent = Gemini(api_key=os.getenv("GEMINI_API_KEY"), llm_prompt_template=llm_prompt_template)
step_back_agent = Gemini(api_key=os.getenv("GEMINI_API_KEY"), llm_prompt_template=step_back_llm_prompt_template)
q_a = []
def pipeline(query: str, is_logging_retrieval: bool = False, is_logging_qa: bool = False):
    # Step back to retrieve context
    rewrite_query = step_back_agent.chain.invoke({"question": query})
    print("Original query: ", query)
    query = rewrite_query
    print("Rewrite query: ", rewrite_query)
    docs = vec_db.retrieve(rewrite_query, is_logging=is_logging_retrieval)
    context_chunks = []
    for doc in docs:
        if doc["type"] == "question":
            context_chunks.append(f"**Chunk: {doc['title']}**\n{doc['content']})")
        else:
            context_chunks.append(f"**Chunk: {doc['title']}**\n{doc['content']}")

    context = "\n".join(context_chunks)
    print("___________CONTEXT___________")
    print(context)
    print("____________________________")
    answer = agent.chain.invoke({"question":query, "context":context})
    if is_logging_qa:
        print({
            # "question": query,
            "answer": answer,
        })
        q_a.append({"question": query, "answer": answer, "context":context, "docs": docs})
    return answer

import streamlit as st
import random
import time

def response_generator(question):
    return pipeline(question)

st.set_page_config(page_title='HCMUTE-Chat', layout='wide', page_icon='./resources/images/icon_1.png')

st.image('./resources/images/icon_2.png')
st.title('TRỢ LÍ ẢO HỖ TRỢ TUYỂN SINH')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Show chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Viết câu hỏi của bạn ở đây?"):
    #Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    #Show user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    #Show assistant response in chat message container
    with st.chat_message("assistant"):
        response_container = st.empty()  
        full_response = ""  
        for word in response_generator(prompt):
            full_response += word
            response_container.markdown(full_response)  
        response = full_response.strip()  

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.expander('Liên hệ với chúng tôi'):
    with st.form(key='contact', clear_on_submit=True):
        email = st.text_input('Email liên hệ của bạn')
        text = st.text_area('Nội dung', '')
        submit_button = st.form_submit_button(label='Gửi thông tin')
        if submit_button:
            with open(f'contacts/{email}.txt', 'wb') as file:
                file.write(text.encode('utf-8'))
            st.success('Thông tin đã được gửi!')
