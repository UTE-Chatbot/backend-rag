from langchain import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
import dotenv

dotenv.load_dotenv()

class BaseLLM:
    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key

class Gemini(BaseLLM):
    def __init__(self, api_key: str, llm_prompt_template: any):
        super().__init__("gemini-1.5-pro", api_key)
        self.llm_prompt = PromptTemplate.from_template(llm_prompt_template)
        self.llm = ChatGoogleGenerativeAI(model=self.name, google_api_key=self.api_key)
        self.chain = self.llm_prompt | self.llm | StrOutputParser()

    def format_prompt(self, question: str, context: str) -> str:
        return self.llm_prompt.format(question=question, context=context)
