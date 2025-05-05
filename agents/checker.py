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
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from fchecker.fscorer import LikertScorer
from agents.schemas import FRPair
from agents.states import FactCheckerState
from pdfparse.rag_llama import RAG
from langchain_core.messages import AIMessage, HumanMessage
import yaml
from agents.schemas import Paper
from pprint import pprint
import gc  # For explicit garbage collection

logging.basicConfig(
    filename="PRAGATI.log",
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


with open("utils/prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

def generate_search_query(claim: str) -> str:
    """Generate a search query from a claim"""
    try:
        prompt_gen = PromptGenerator()
        query = prompt_gen.generate_prompt(
            "tavily", claim, search_depth="normal", max_results=5
        )
        return query.strip()
    except Exception as e:
        logger.error(f"Error generating search query: {e}")
        return claim


def academic_search(query: str) -> List[Dict[str, Any]]:
    """Focus search on academic sources like ArXiv and Google Scholar"""
    results = []
    
    # Limited to 3 results per source to reduce memory usage
    max_results_per_source = 3
    
    for tool in [ArixvSearchTool(), GoogleScholarSearchTool()]:
        try:
            tool_results = tool.invoke_tool(query)
            if tool_results:
                # Process only essential information
                for result in tool_results[:max_results_per_source]:
                    # Store only critical fields to reduce memory usage
                    compact_result = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source": tool.__class__.__name__,
                    }
                    
                    # Keep only a snippet of content, not the full text
                    if "content" in result and result["content"]:
                        compact_result["content"] = result["content"][:500]  # Limit content size
                    elif "snippet" in result and result["snippet"]:
                        compact_result["snippet"] = result["snippet"][:500]  # Limit snippet size
                        
                    results.append(compact_result)
        except Exception as e:
            logger.warning(f"Error with {tool.__class__.__name__}: {e}")
        
        # Clean up after each tool to free memory
        gc.collect()

    return results[:5]  # Return at most 5 results


def general_search(query: str) -> List[Dict[str, Any]]:
    """Use only general web search for broad topics"""
    try:
        tool = TavilySearchTool()
        tool_msg = tool.invoke_tool(query)
        results = json.loads(tool_msg.content)
        
        compact_results = []
        if results:
            # Process only top 5 results with limited content
            for result in results[:5]:
                if isinstance(result, dict):
                    compact_result = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source": tool.__class__.__name__,
                    }
                    
                    # Keep only a snippet of content, not the full text
                    if "content" in result and result["content"]:
                        compact_result["content"] = result["content"][:500]  # Limit content size
                    elif "snippet" in result and result["snippet"]:
                        compact_result["snippet"] = result["snippet"][:500]  # Limit snippet size
                        
                    compact_results.append(compact_result)
            
            return compact_results
        return []
    except (Exception, json.JSONDecodeError) as e:
        logger.warning(f"Error with {tool.__class__.__name__}: {e}")
        return []
    finally:
        # Clean up to free memory
        gc.collect()


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
        return {"score": 0}

    # Score the claim
    result, token = scorer.score_text(claim, combined_references)
    return {"score": result, "token": token}


def parse_claims(state: FactCheckerState) -> FactCheckerState:
    """Parse the input text into separate claims to check"""
    try:
        # For simplicity, we'll treat each line as a separate claim
        claims = [
            line.strip() for line in state.inputs.strip().split("\n") if line.strip()
        ]

        if not claims:
            state.errors = "No valid claims found in input"
            return state

        # Initialize pairs with claims - only store claims, not search results yet
        state.pairs = [FRPair(claim=claim) for claim in claims]
        state.no_claims = len(state.pairs)
        
        # Pre-allocate scores array for memory efficiency
        state.claim_scores = [0] * state.no_claims
        
        logger.info(f"Parsed {len(state.pairs)} claims from input")
    except Exception as e:
        state.errors = f"Error parsing claims: {str(e)}"
        logger.error(f"Error in parse_claims: {e}")
    finally:
        # Clean up to free memory
        gc.collect()

    return state

def retrieve_facts(state: FactCheckerState) -> FactCheckerState:
    """Retrieve facts from the knowledge base"""
    file = state.paper.filepath
    if not file:
        state.errors = "No file provided for knowledge base"
        return state
    
    try:
        logger.info(f"Starting RAG application with PDF: {file}")
        rag = RAG(file)
        index = rag.create_db()
        query_text = prompts["fchecker"]["human_message"].format(paper_name=state.paper.title)
        retriever = rag.create_retriever(index)
        response = rag.rag_query(query_text, retriever)
        # pprint(response["result"])
        state.inputs = response["result"]
    except Exception as e:
        state.errors = f"Error retrieving facts: {str(e)}"
        logger.error(f"Error in retrieve_facts: {e}")
    finally:
        # Clean up to free memory
        gc.collect()
    
    return state

def generate_query(state: FactCheckerState) -> FactCheckerState:
    """Generate search queries for the current claim"""
    if state.current_index >= len(state.pairs):
        return state

    try:
        current_claim = state.pairs[state.current_index].claim
        logger.info(f"Generating query for claim: {current_claim}")
        query = generate_search_query(current_claim)
        
        # Store query directly in the current pair
        state.pairs[state.current_index].search_query = query
        logger.info(f"Generated query: {query}")
    except Exception as e:
        state.errors = f"Error generating query: {str(e)}"
        logger.error(f"Error in generate_query: {e}")
    finally:
        # Clean up to free memory
        gc.collect()

    return state


def search_web(state: FactCheckerState) -> FactCheckerState:
    """Perform web search using the generated query"""
    if state.current_index >= len(state.pairs) or state.errors:
        return state

    try:
        current_pair = state.pairs[state.current_index]
        if not current_pair.search_query:
            state.errors = "Missing search query"
            return state

        logger.info(f"Searching for: {current_pair.search_query}")

        # Use the appropriate search function based on claim content
        if any(
            term in current_pair.claim.lower()
            for term in ["research", "study", "scientific", "paper"]
        ):
            search_results = academic_search(current_pair.search_query)
        else:
            search_results = general_search(current_pair.search_query)

        # Store results directly in the current pair
        current_pair.search_results = search_results
        logger.info(f"Found {len(search_results)} results")
    except Exception as e:
        state.errors = f"Error in web search: {str(e)}"
        logger.error(f"Error in search_web: {e}")
    finally:
        # Clean up to free memory
        gc.collect()

    return state


def verify_claim(state: FactCheckerState) -> FactCheckerState:
    """Verify the current claim against search results"""
    if state.current_index >= len(state.pairs) or state.errors:
        return state
    
    try:
        current_pair = state.pairs[state.current_index]
        logger.info(f"Verifying claim: {current_pair.claim}")
        
        # Score the claim
        verification = score_fact(
            claim=current_pair.claim, 
            references=current_pair.search_results
        )
        
        # Store only the verification score, not the full results
        score = verification.get("score", 0)
        state.claim_scores[state.current_index] = score
        
        # Store minimal verification info
        current_pair.verification = {"score": score}
        
        if score == 0:
            logger.info(f"Claim is Unverified")
        else:
            logger.info(f"Verification complete: Score={score}")
        
        # Memory optimization: Clear search results after verification
        current_pair.search_results = []
            
        # Add to total score
        state.total_score += score
        
    except Exception as e:
        state.errors = f"Error in verification: {str(e)}"
        logger.error(f"Error in verify_claim: {e}")
    
    # Always increment the index and calculate average
    state.current_index += 1
    
    # Calculate average score
    if state.current_index <= state.no_claims:
        state.average_score = state.total_score / state.current_index
    else:
        state.average_score = state.total_score / state.no_claims
        
    # Determine if the paper is factual
    state.is_factual = state.average_score > 3
    
    # Force garbage collection
    gc.collect()
        
    return state

def should_continue(state: FactCheckerState) -> str:
    """Determine whether to process the next claim or end"""
    if state.errors:
        logger.error(f"Ending workflow due to error: {state.errors}")
        return END
    if state.current_index >= state.no_claims:
        logger.info("All claims processed, ending workflow")
        logger.info(f"Average score: {state.average_score}")
        logger.info(f"Is factual: {state.is_factual}")
        return END

    logger.info(f"Moving to next claim ({state.current_index+1}/{state.no_claims})")
    return "generate_query"


def create_fact_checker_graph() -> StateGraph:
    """Create the fact checker workflow graph"""
    workflow = StateGraph(FactCheckerState)
    workflow.add_node("paper_parser", retrieve_facts)
    workflow.add_node("parse_claims", parse_claims)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("search_web", search_web)
    workflow.add_node("verify_claim", verify_claim)

    # Define the edges
    workflow.add_edge(START, "paper_parser")
    workflow.add_edge("paper_parser", "parse_claims")
    workflow.add_edge("parse_claims", "generate_query")
    workflow.add_edge("generate_query", "search_web")
    workflow.add_edge("search_web", "verify_claim")
    workflow.add_conditional_edges("verify_claim", should_continue)
    
    # Compile the graph
    compiled_graph = workflow.compile()
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fact_checker_graph.png")
        graph_image = compiled_graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
        with open(output_path, "wb") as f:
            f.write(graph_image)
        logger.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")

    return compiled_graph


def fact_check(paper: Paper) -> List[FRPair]:
    """Run the fact checking process on a text input"""
    logger.info("Starting fact checking process")
    checker = create_fact_checker_graph()
    final_state = checker.invoke({"paper": paper}, {"recursion_limit": 1000})

    # Verify we have the scores
    for i, score in enumerate(final_state["claim_scores"]):
        final_state["pairs"][i].verification = {"score": score}

    return final_state["pairs"]


def format_results(pairs: List[FRPair]) -> str:
    """Format the fact checking results into a readable string"""
    results = []
    for i, pair in enumerate(pairs):
        result = f"Claim {i+1}: {pair.claim}\n"
        if hasattr(pair, "verification") and pair.verification:
            v = pair.verification
            if v.get("score", "N/A") == 0:
                result += "Claim is Unverified\n"
            else:
                result += f"Score: {v.get('score', 'N/A')}/5\n"
        else:
            result += "Verification failed\n"
        results.append(result)

    return "\n" + "-" * 50 + "\n".join(results)


# Example usage
if __name__ == "__main__":
    paper1 = Paper(
            filepath="/home/naba/Desktop/PRAGATI/satya.pdf",
            title="SATYA",
            topic="Agentic AI SYSTEM",
            filename="SATYA.pdf",
            sections=["Abstract", "Introduction", "Methodology"]
    )
    results = fact_check(paper1)