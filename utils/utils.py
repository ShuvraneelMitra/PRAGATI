import re
from agents.states import QuestionState, AnswerState


def compile_questions(state: QuestionState) -> AnswerState:
    """
    Compile all questions made by all the reviewers into a single list
    """
    questions = []
    for reviewer in state.reviewers:
        questions.extend(reviewer.questions)

    updated_state = AnswerState(
        messages=state.messages,
        paper=state.paper,
        questions=questions,
    )

    return updated_state


def tmpl_to_prompt(text: str, mapping: dict) -> str:
    """
    Converts a string template with variables described as
    `${var}`, into a string with those values replaced.
    Args:
        text: the template in the form of a string
        mapping: The dictionary containing the key-value pairs
                    regarding the replacements to be made.

    Returns: the text with the replacements made.
    """

    def replace_fn(match: re.Match) -> str:
        key = match.group(1)
        if key not in mapping.keys():
            return "undefined"
        else:
            return mapping[key]

    return re.sub(r"\$\{(.*?)\}", replace_fn, text)
