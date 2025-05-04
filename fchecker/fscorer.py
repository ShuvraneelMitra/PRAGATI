import os
import yaml
import re
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
        response, i, o = invoke_llm_langchain(messages)
        content = response[-1].content.strip()
        score_match = re.match(r'^\s*(\d+)', content)
        if score_match:
            score = int(score_match.group(1))
        else:
            numbers = re.findall(r'\d+', content)
            if numbers:
                score = int(numbers[0])
            else:
                score = 0  
        return score, int(i + o)

# Example usage
if __name__ == "__main__":
    scorer = LikertScorer()
    text = "sun rises in the west and it's very far from earth"
    fact = """The Sun, the Moon, the planets, and the stars all rise in the east and set in the west. And that's because Earth spins -- toward the east."""
    score, tokens = scorer.score_text(text, fact)
    print("Likert Score:", score)