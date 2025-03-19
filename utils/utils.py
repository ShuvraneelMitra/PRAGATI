import re
from agents.states import QuestionState


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
