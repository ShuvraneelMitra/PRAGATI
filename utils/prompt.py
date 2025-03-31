import yaml
from typing import Dict, List, Any
from utils.chat import invoke_llm_langchain
import os
import json
from langchain_core.messages import AIMessage, HumanMessage


class PromptGenerator:
    """
    A class that generates structured prompts for various search engines like Tavily, arXiv, etc.
    """

    def __init__(
        self, config_path: str = "config_web.yaml", prompts_path: str = "prompts.yaml"
    ):

        config_path = os.path.join(os.path.dirname(__file__), "config_web.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")

        with open(prompts_path, "r") as file:
            self.prompts = yaml.safe_load(file)["web_prompts"]

        self.initial_message = AIMessage(content=self.prompts["initial_message"])

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
            raise ValueError(
                f"Unsupported search provider: {provider}. "
                f"Supported providers: {list(self.search_providers.keys())}"
            )

        latest_message = HumanMessage(
            content=f"I want to get a search query for the topic: {query} "
            f"which is searched on {provider}"
        )

        structured_query, _, _ = invoke_llm_langchain(
            [self.initial_message, latest_message]
        )

        # Ensure exclude_domains and include_domains are always passed
        if provider == "tavily":
            kwargs.setdefault("include_domains", [])
            kwargs.setdefault("exclude_domains", [])
        elif provider == "arxiv":
            kwargs.setdefault("categories", [])
            kwargs.setdefault("sort_by", "relevance")
            kwargs.setdefault("sort_order", "descending")
        elif provider == "scholar":
            kwargs.setdefault("year_start", None)
            kwargs.setdefault("year_end", None)
            kwargs.setdefault("include_citations", False)

        formatted_prompt = self.search_providers[provider](
            structured_query[-1], **kwargs
        )
        # Parse the returned query string as JSON if it is in JSON format
        try:
            if isinstance(formatted_prompt["query"], str):
                # Try to parse as JSON if the query is a JSON string
                json_query = json.loads(formatted_prompt["query"])
                formatted_prompt["query"] = json_query
        except json.JSONDecodeError:
            pass
        return formatted_prompt["query"].content[9:-1]

    def _tavily_prompt_template(
        self,
        query: str,
        search_depth: str,
        max_results: int,
        include_domains: List[str],
        exclude_domains: List[str],
    ) -> Dict[str, Any]:
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

    def _arxiv_prompt_template(
        self,
        query: str,
        max_results: int,
        categories: List[str],
        sort_by: str,
        sort_order: str,
    ) -> Dict[str, Any]:
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

    def _scholar_prompt_template(
        self,
        query: str,
        max_results: int,
        year_start: int,
        year_end: int,
        include_citations: bool,
    ) -> Dict[str, Any]:
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
            include_domains=["nature.com", "science.org"],
        )
        print(tavily_prompt)

        arxiv_prompt = prompt_gen.generate_prompt(
            "arxiv",
            "Transformer models for computer vision",
            max_results=5,
            categories=["cs.CV", "cs.LG"],
            sort_by="relevance",
        )
        print(arxiv_prompt)

        scholar_prompt = prompt_gen.generate_prompt(
            "scholar",
            "Climate change impact on agriculture",
            max_results=7,
            year_start=2020,
            include_citations=True,
        )
        print(scholar_prompt)


if __name__ == "__main__":
    PromptGenerator.sample_usage()
