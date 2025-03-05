from rag.vector_store import init_vectorstore
from embeddings.jina import JinaEmbedding as Jina
import os
from models.gemini import Gemini

# INIT MODELS
embedding_model = Jina()

llm_prompt_template ="""
Hãy trở thành một trợ lý ảo của trường Đại học Sư phạm Kỹ thuật TP.HCM. Bạn sẽ trả lời các câu hỏi của sinh viên về trường học. Là người đại diện của trường, bạn cần trả lời chính xác từ ác thông tin được cung cấp.
        Hãy đóng vai trò của mình và trả lời câu hỏi sau:
        Câu hỏi: {question}
        Ngữ cảnh: {context}
        Trả lời:
    """


vec_db = init_vectorstore(data_path="./data/V1_DATASET.csv", db_name="hcmute-admission-v1", embed_columns=["title", "content"], embedding_model=embedding_model,rerank_top_k=5)
agent = Gemini(api_key=os.getenv("GEMINI_API_KEY"), llm_prompt_template=llm_prompt_template)


q_a = []
def pipeline(query: str, is_logging_retrieval: bool = False, is_logging_qa: bool = False):
    docs = vec_db.retrieve(query, is_logging=is_logging_retrieval)
    context = "\n".join([doc["title"] + "; " + doc["content"] for doc in docs])
    answer = agent.chain.invoke({"question":query, "context":context})
    if is_logging_qa:
        q_a.append({"question": query, "answer": answer, "context":context, "docs": docs})
    return answer

# # print(pipeline("Lịch sử hình thành và phát triển HCMUTE?", is_logging_qa=True))
# # print(pipeline("trường mình có làm việc t7 k ạ", is_logging_qa=True))
# # print(pipeline("Cho em hỏi ngành công nghệ kỹ thuật ô tô là mình học về ô tô điện hay là mình học về ô tô truyền thống hay là học cả 2 ạ", is_logging_qa=True))
# # print(pipeline("Nhà gửi xe trường mình hay kẹt không ạ", is_logging_qa=True))
# # print(pipeline("Trường có học bổng không?", is_logging_qa=True))
# # print(pipeline("Trường có ký túc xá không?", is_logging_qa=True))
# # print(pipeline("Trường có học phí không?", is_logging_qa=True))
# # print(pipeline("Ngành của trường", is_logging_qa=True))
# # print(pipeline("Nhuộm tóc được k v?", is_logging_qa=True))
# print(pipeline("Trường thành lập bao lâu rồi?", is_logging_qa=True))


# UI

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
