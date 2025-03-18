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
    def __init__(self, api_key: str, llm_prompt_template: any, model_variant="gemini-2.0-pro-exp-02-05"):
        super().__init__(model_variant, api_key)
        self.llm_prompt = llm_prompt_template

        print(model_variant)
        self.llm = ChatGoogleGenerativeAI(model=model_variant, google_api_key=self.api_key)
        self.chain = self.llm_prompt | self.llm | StrOutputParser()

    def format_prompt(self, question: str, context: str) -> str:
        return self.llm_prompt.format(question=question, context=context)
