from agents.answer import agen_graph
from agents.persona import qgen_graph
from agents.schemas import TokenTracker, QAPair, Reviewer, Paper, FRPair
from agents.states import QuestionState, SingleQuery
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename="PRAGATI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_combined_graph(qstate: QuestionState):
    qgraph = qgen_graph()
    q_output = qgraph.invoke(qstate)

    agraph = agen_graph()
    a_output = agraph.invoke(q_output)

    logger.info(a_output)

if __name__ == "__main__":
    ex = {
        "messages": [],
        "paper": Paper(
            filepath="/home/naba/Desktop/PRAGATI/mrinmoy.pdf",
            topic="Do we really need Foundation Models for multi-step-ahead Epidemic Forecasting?",
            sections=["Introduction", "Methodology", "Conclusion"],
        ),
        "num_reviewers": 1,
        "token_usage": TokenTracker(
            net_input_tokens=0, net_output_tokens=0, net_tokens=0
        ),
        "reviewers": [],
        "queries": [],
    }

    qstate = QuestionState(**ex)
    run_combined_graph(qstate)
