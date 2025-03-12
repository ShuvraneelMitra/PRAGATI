import os
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from utils.chat import invoke_llm_langchain

class LikertScorer:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) 
        prompts_path = os.path.join(project_root, "utils", "prompts.yaml")
        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["scorer_prompts"]

    def score_text(self, text, fact):
        initial_message = self.prompts["initial_message"].format(text=text, fact=fact) 
        messages = [HumanMessage(content=initial_message)]
        response, _, _ = invoke_llm_langchain(messages)
        return response[-1].content.strip()

# Example usage
if __name__ == "__main__":
    scorer = LikertScorer()
    text = "sun rises in the west and it's very far from earth"
    fact = "sun rises in the east\n the distance of sun from earth is 148.63 million km"
    
    score = scorer.score_text(text, fact)
    print("Likert Score:", score)
