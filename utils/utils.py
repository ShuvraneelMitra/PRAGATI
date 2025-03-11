from agents.states import QuestionState, AnswerState


def compile_questions(state: QuestionState) -> AnswerState:
    """
    Compile all questions made by all the reviewers into a single list
    """
    questions = []
    for reviewer in state.reviewers:
        questions.extend(reviewer.questions)

    updated_state = AnswerState(messages=state.messages,
                                paper=state.paper,
                                questions=questions,
                                )

    return updated_state
