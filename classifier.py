from data.loader import load_data, load_unique_categories
import os

from models.gemini import Gemini
from langchain import PromptTemplate

class CategoryClassifier():
    def __init__(self):
        self.categories = load_unique_categories("./data/V2_DATASET.csv", "category")
        llm_prompt_template = PromptTemplate(
            template=""""  
                Bạn là một trợ lý chuyên nghiệp tuyển sinh của trường đại học sư phạm kỹ thuật Tp. Hồ Chí Minh. Bạn được cung cấp các thẻ phân loại sau: {categories}. Hãy phân loại câu hỏi {question} thành 1 trong các thẻ trên và chỉ đưa ra duy nhất giá trị của thẻ mà bạn phân loại. 
                ***Lưu ý***: Thẻ DIEM_CHUAN_CAC_NAM là thẻ phân loại về điểm, điểm chuẩn các năm trong bộ dữ liệu.
                Thẻ phân loại là:""",
            input_variables=["categories", "question"]
        )
        agent = Gemini(api_key=os.getenv("GEMINI_API_KEY"), llm_prompt_template=llm_prompt_template, model_variant="gemini-1.5-pro")
        self.classifier_agent = agent
    def classify(self, query):
        return self.classifier_agent.chain.invoke({"question": query, "categories": self.categories})
