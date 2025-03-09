from typing import Dict, List, Any
from chat import invoke_llm_langchain
from langchain_core.messages import AIMessage, HumanMessage

class PromptGenerator:
    """
    A class that generates structured prompts for various search engines like Tavily, arXiv, etc.
    """
    def __init__(self):
        self.initial_message = AIMessage(content=(
            "Generate only one well-structured and precise web search query based on user intent. ",
            "the query should be one sentence long and should be clear and concise. and don't say anthing else in the query that is not relevant to the search. ",
            "Ensure clarity by removing ambiguity, optimizing for relevant keywords, and structuring it for efficient search engine results. ",
            "Adapt the query style based on context (e.g., informational, navigational, transactional) and refine phrasing to maximize accuracy and relevance.",
            "dont't put anthing in the query that is not relevant to the search",
        ))
        self.search_providers = {
            "tavily": self._tavily_prompt_template,
            "arxiv": self._arxiv_prompt_template,
            "scholar": self._scholar_prompt_template,
        }

    def generate_prompt(self, provider: str, query: str, **kwargs) -> str:
        """
        Generate a structured prompt for the specified search provider.
        """
        provider = provider.lower()
        if provider not in self.search_providers:
            raise ValueError(f"Unsupported search provider: {provider}. "
                             f"Supported providers: {list(self.search_providers.keys())}")

        latest_message = HumanMessage(content=f"I want to get a search query for the topic: {query} "
                                              f"which is searched on {provider}")
        
        structured_query, _, _ = invoke_llm_langchain([self.initial_message, latest_message])

        formatted_prompt = self.search_providers[provider](structured_query[-1], **kwargs)

        return f"{provider.capitalize()} Prompt: {formatted_prompt}"

    def _tavily_prompt_template(self, query: str, search_depth: str = "moderate", 
                                max_results: int = 5, include_domains: List[str] = None,
                                exclude_domains: List[str] = None) -> Dict[str, Any]:
        """Generate a prompt structure for Tavily search"""
        prompt = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }
        if include_domains:
            prompt["include_domains"] = include_domains
        if exclude_domains:
            prompt["exclude_domains"] = exclude_domains
        return prompt

    def _arxiv_prompt_template(self, query: str, max_results: int = 5, 
                               categories: List[str] = None, 
                               sort_by: str = "relevance", 
                               sort_order: str = "descending") -> Dict[str, Any]:
        """Generate a prompt structure for arXiv search"""
        prompt = {
            "query": query,
            "max_results": max_results,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }
        if categories:
            prompt["categories"] = categories
        return prompt

    def _scholar_prompt_template(self, query: str, max_results: int = 5,
                                 year_start: int = None, year_end: int = None,
                                 include_citations: bool = False) -> Dict[str, Any]:
        """Generate a prompt structure for Google Scholar search"""
        prompt = {
            "query": query,
            "max_results": max_results,
            "include_citations": include_citations,
        }
        if year_start:
            prompt["year_start"] = year_start
        if year_end:
            prompt["year_end"] = year_end
        return prompt

    @staticmethod
    def sample_usage():
        """Demonstrates how to use the PromptGenerator class"""
        prompt_gen = PromptGenerator()

        tavily_prompt = prompt_gen.generate_prompt(
            "tavily", 
            "Latest developments in quantum computing",
            search_depth="deep",
            max_results=10,
            include_domains=["nature.com", "science.org"]
        )
        print(tavily_prompt)

        arxiv_prompt = prompt_gen.generate_prompt(
            "arxiv", 
            "Transformer models for computer vision",
            max_results=5,
            categories=["cs.CV", "cs.LG"],
            sort_by="relevance"
        )
        print(arxiv_prompt)

        scholar_prompt = prompt_gen.generate_prompt(
            "scholar", 
            "Climate change impact on agriculture",
            max_results=7,
            year_start=2020,
            include_citations=True
        )
        print(scholar_prompt)

if __name__ == "__main__":
    PromptGenerator.sample_usage()
