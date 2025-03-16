from models.gemini import Gemini

class AgentLLM():
    def __init__(self, agent:Gemini, agent_prompt):
        self.agent = agent
        self.agent.llm_prompt = agent_prompt

    def invoke(self, params):
        return self.agent.invoke(params)
