from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from dotenv import load_dotenv
import json

class TavilySearchTool:
    def __init__(self, max_results=5, search_depth="advanced", include_answer=True, 
                 include_raw_content=True, include_images=True):
        load_dotenv()
        self.tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
    
    def invoke_tool(self, query, tool_id="1", tool_name="tavily", tool_type="tool_call"):
        model_generated_tool_call = {
            "args": {"query": query},
            "id": tool_id,
            "name": tool_name,
            "type": tool_type,
        }
        return self.tool.invoke(model_generated_tool_call)

class ArixvSearchTool:
    def __init__(self, max_results=5, document_length=4):
        load_dotenv()
        self.arxiv = ArxivAPIWrapper(
            top_k_results=max_results,
            doc_content_chars_max=document_length * 1000
        )
        
    def invoke_tool(self, query):
        """
        Search for papers on Arxiv using the provided query.
        
        Args:
            query (str): The search query.
            
        Returns:
            list: A list of papers matching the query.
        """
        try:
            results = self.arxiv.run(query)
            return results
        except Exception as e:
            return f"Error searching Arxiv: {str(e)}"

class GoogleScholarSearchTool:
    def __init__(self):
        load_dotenv()
        self.scholar = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
    
    def invoke_tool(self, query):
        """
        Search for scholarly articles using Google Scholar.
        
        Args:
            query (str): The search query.
            
        Returns:
            str: Results from Google Scholar search.
        """
        try:
            results = self.scholar.run(query)
            return results
        except Exception as e:
            return f"Error searching Google Scholar: {str(e)}"


if __name__ == "__main__":
    # Test Tavily search
    search_tool = TavilySearchTool()
    tool_msg = search_tool.invoke_tool("sun rises in the west")
    try:
        results = json.loads(tool_msg.content)  
        for result in results:
            if isinstance(result, dict):  
                title = result.get("title", "No Title")
                url = result.get("url", "No URL")
                content = result.get("content", "No Content")
                print(f"Title: {title}\nURL: {url}\nContent: {content}\n")
            else:
                print(f"Unexpected result format: {result}")
    except json.JSONDecodeError:
        print("Error: Could not parse JSON content.") 
    # Test Arxiv search
    print("\n--- Arxiv Search Test ---")
    arxiv_tool = ArixvSearchTool()
    arxiv_results = arxiv_tool.invoke_tool("quantum computing applications")
    if isinstance(arxiv_results, str) and arxiv_results.startswith("Error"):
        print(arxiv_results)
    else:
        print(f"Found {len(arxiv_results.split('Title:')) - 1} Arxiv papers")
        print(arxiv_results[:500] + "..." if len(arxiv_results) > 500 else arxiv_results)
    
    # Test Google Scholar search
    print("\n--- Google Scholar Search Test ---")
    scholar_tool = GoogleScholarSearchTool()
    scholar_results = scholar_tool.invoke_tool("machine learning advancements")
    if isinstance(scholar_results, str) and scholar_results.startswith("Error"):
        print(scholar_results)
    else:
        print(scholar_results[:500] + "..." if len(scholar_results) > 500 else scholar_results)