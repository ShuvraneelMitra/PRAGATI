from langchain_core.messages import AIMessage, HumanMessage
from utils.chat import invoke_llm_langchain

class LikertScorer:
    def __init__(self):
        pass

    def score_text(self, text, fact):
        messages = [
            HumanMessage(content=f"""
            You are a fact-checking AI. Given the following text and fact, evaluate the correctness of the text on a Likert scale (1 to 5):
            
            Text: "{text}"
            
            Fact: "{fact}"
            
            Likert Scale:
            1 - Completely Incorrect
            2 - Mostly Incorrect
            3 - Partially Correct
            4 - Mostly Correct
            5 - Completely Correct
            
            Provide only the score as output.
            """)
        ]
        
        response, _, _ = invoke_llm_langchain(messages)
        return response[-1].content.strip()

# Example usage
if __name__ == "__main__":
    scorer = LikertScorer()
    text = "sun rises in the west and it's very far from earth"
    fact = "sun rises in the east\n the distance of sun from earth is 148.63 million km"
    
    score = scorer.score_text(text, fact)
    print("Likert Score:", score)