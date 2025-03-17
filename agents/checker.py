from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import json
import sys
import os
import logging
from PIL import Image
from utils.prompt import PromptGenerator
from fchecker.webs import TavilySearchTool, ArixvSearchTool, GoogleScholarSearchTool
from fchecker.fscorer import LikertScorer
from agents.schemas import FRPair
from agents.states import TokenTracker, FactCheckerState
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_search_query(claim: str) -> str:
    """Generate a search query from a claim"""
    try:
        prompt_gen = PromptGenerator()
        query = prompt_gen.generate_prompt(
            "tavily", 
            claim,
            search_depth="normal",
            max_results=5
        )
        return query.strip()
    except Exception as e:
        logger.error(f"Error generating search query: {e}")
        return claim 

def academic_search(query: str) -> List[Dict[str, Any]]:
    """Focus search on academic sources like ArXiv and Google Scholar"""
    results = []
    for tool in [ArixvSearchTool(), GoogleScholarSearchTool()]:
        try:
            tool_results = tool.search(query)
            if tool_results:
                for result in tool_results:
                    result["source"] = tool.__class__.__name__
                results.extend(tool_results)
        except Exception as e:
            logger.warning(f"Error with {tool.__class__.__name__}: {e}")
    
    return results[:5]  

def general_search(query: str) -> List[Dict[str, Any]]:
    """Use only general web search for broad topics"""
    try:
        tool = TavilySearchTool()
        tool_msg = tool.invoke_tool(query)
        results = json.loads(tool_msg.content)
        if results:
            for result in results:
                if isinstance(result, dict):
                    result["source"] = tool.__class__.__name__
            return results[:5]  # Return top 5 results
        return []
    except (Exception, json.JSONDecodeError) as e:
        logger.warning(f"Error with {tool.__class__.__name__}: {e}")
        return []

def score_fact(claim: str, references: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score the claim against references"""
    scorer = LikertScorer()
    reference_texts = []
    for ref in references:
        content = ""
        if "content" in ref and ref["content"]:
            content = ref["content"]
        elif "snippet" in ref and ref["snippet"]:
            content = ref["snippet"]
        
        if content:
            source = ref.get("source", "Unknown")
            url = ref.get("url", "")
            reference_texts.append(f"Source ({source}): {content} [URL: {url}]")
    
    combined_references = "\n\n".join(reference_texts)
    
    if not combined_references:
        return {
            "score": 0
        }
    
    # Score the claim
    result = scorer.score_text(claim, combined_references)
    return {
        "score": result
    }

def parse_claims(state: FactCheckerState) -> FactCheckerState:
    """Parse the input text into separate claims to check"""
    try:
        # For simplicity, we'll treat each line as a separate claim
        claims = [line.strip() for line in state.inputs.strip().split("\n") if line.strip()]
        
        if not claims:
            state.errors = "No valid claims found in input"
            return state
        
        # Initialize pairs with claims
        state.pairs = [FRPair(claim=claim) for claim in claims]
        logger.info(f"Parsed {len(state.pairs)} claims from input")
    except Exception as e:
        state.errors = f"Error parsing claims: {str(e)}"
        logger.error(f"Error in parse_claims: {e}")
    
    return state

def generate_query(state: FactCheckerState) -> FactCheckerState:
    """Generate search queries for the current claim"""
    if state.current_index >= len(state.pairs):
        return state
    
    current_pair = state.pairs[state.current_index]
    try:
        logger.info(f"Generating query for claim: {current_pair.claim}")
        query = generate_search_query(current_pair.claim)
        current_pair.search_query = query
        state.pairs[state.current_index] = current_pair
        logger.info(f"Generated query: {query}")
    except Exception as e:
        state.errors = f"Error generating query: {str(e)}"
        logger.error(f"Error in generate_query: {e}")
    
    return state

def search_web(state: FactCheckerState) -> FactCheckerState:
    """Perform web search using the generated query"""
    if state.current_index >= len(state.pairs) or state.errors:
        return state
    
    current_pair = state.pairs[state.current_index]
    try:
        if not current_pair.search_query:
            state.errors = "Missing search query"
            return state
            
        logger.info(f"Searching for: {current_pair.search_query}")
        
        # Use the appropriate search function based on claim content
        if any(term in current_pair.claim.lower() for term in ["research", "study", "scientific", "paper"]):
            search_results = academic_search(current_pair.search_query)
        else:
            search_results = general_search(current_pair.search_query)
            
        current_pair.search_results = search_results
        state.pairs[state.current_index] = current_pair
        logger.info(f"Found {len(search_results)} results")
    except Exception as e:
        state.errors = f"Error in web search: {str(e)}"
        logger.error(f"Error in search_web: {e}")
    
    return state

def verify_claim(state: FactCheckerState) -> FactCheckerState:
    """Verify the current claim against search results"""
    if state.current_index >= len(state.pairs) or state.errors:
        return state
    
    current_pair = state.pairs[state.current_index]
    try:
        logger.info(f"Verifying claim: {current_pair.claim}")
        verification = score_fact(
            claim=current_pair.claim, 
            references=current_pair.search_results
        )
        current_pair.verification = verification
        state.pairs[state.current_index] = current_pair
        if(verification.get('score', 'N/A') == 0):
            logger.info(f"Claim is Unverified")
        else:
            logger.info(f"Verification complete: Score={verification.get('score')}")
    except Exception as e:
        state.errors = f"Error in verification: {str(e)}"
        logger.error(f"Error in verify_claim: {e}")
    
    return state

def should_continue(state: FactCheckerState) -> str:
    """Determine whether to process the next claim or end"""
    if state.errors:
        logger.error(f"Ending workflow due to error: {state.errors}")
        return END
    
    state.current_index += 1
    if state.current_index >= len(state.pairs):
        logger.info("All claims processed, ending workflow")
        return END
    
    logger.info(f"Moving to next claim ({state.current_index+1}/{len(state.pairs)})")
    return "generate_query"

def create_fact_checker_graph() -> StateGraph:
    """Create the fact checker workflow graph"""
    #defining the nodes
    workflow = StateGraph(FactCheckerState)
    workflow.add_node("parse_claims", parse_claims)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("search_web", search_web)
    workflow.add_node("verify_claim", verify_claim)
    
    # Define the edges
    workflow.add_edge(START, "parse_claims")
    workflow.add_edge("parse_claims", "generate_query")
    workflow.add_edge("generate_query", "search_web")
    workflow.add_edge("search_web", "verify_claim")
    workflow.add_edge("verify_claim", END)
    
    # Compile the graph
    compiled_graph = workflow.compile()
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        # Save the graph visualization
        output_path = os.path.join(output_dir, "fact_checker_graph.png")
        workflow.draw(output_file=output_path)
        logger.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")
    
    return compiled_graph

def fact_check(text: str) -> List[FRPair]:
    """Run the fact checking process on a text input"""
    logger.info("Starting fact checking process")
    checker = create_fact_checker_graph()
    final_state = checker.invoke({"inputs": text})
    
    if final_state.errors:
        logger.error(f"Fact checking completed with errors: {final_state.errors}")
    else:
        logger.info(f"Fact checking completed successfully for {len(final_state.pairs)} claims")
    
    return final_state.pairs

def format_results(pairs: List[FRPair]) -> str:
    """Format the fact checking results into a readable string"""
    results = []
    for i, pair in enumerate(pairs):
        result = f"Claim {i+1}: {pair.claim}\n"
        if hasattr(pair, "verification") and pair.verification:
            v = pair.verification
            if(v.get('score', 'N/A') == 0):
                result += "Claim is Unverified\n"
            else:
                result += f"Score: {v.get('score', 'N/A')}/5\n"
        else:
            result += "Verification failed\n"
        results.append(result)
    
    return "\n" + "-"*50 + "\n".join(results)

# Example usage
if __name__ == "__main__":
    sample_text = "The Earth is flat.\nWater boils at 100 degrees Celsius at sea level."
    results = fact_check(sample_text)
    
    for i, pair in enumerate(results):
        print(f"Claim {i+1}: {pair.claim}")
        print(f"Verification: {json.dumps(pair.verification, indent=2)}")
        print("-" * 50)