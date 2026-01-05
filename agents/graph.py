from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .state import PriceComparisonState
from .nodes import (
    price_prediction_node,
    web_search_node,
    scraping_node,
    final_analysis_node
)
from .tools import search_web_tool, scrape_page_tool

def should_continue_to_search(state: PriceComparisonState) -> str:
    """Decide whether to continue to web search based on prediction success"""
    predicted_price = state.get("predicted_price")
    if predicted_price is not None:
        return "web_search"
    else:
        return "final_analysis"

def should_continue_to_scrape(state: PriceComparisonState) -> str:
    """Decide whether to continue to scraping based on search results"""
    search_results = state.get("search_results", [])
    if search_results:
        return "scraping"
    else:
        return "final_analysis"

def create_price_comparison_graph():
    workflow = StateGraph(PriceComparisonState)
    
    # Add nodes
    workflow.add_node("predict_price", price_prediction_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("scraping", scraping_node)
    workflow.add_node("final_analysis", final_analysis_node)
    
    # Set entry point
    workflow.set_entry_point("predict_price")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "predict_price",
        should_continue_to_search,
        {
            "web_search": "web_search",
            "final_analysis": "final_analysis"
        }
    )
    
    workflow.add_conditional_edges(
        "web_search",
        should_continue_to_scrape,
        {
            "scraping": "scraping",
            "final_analysis": "final_analysis"
        }
    )
    
    workflow.add_edge("scraping", "final_analysis")
    workflow.add_edge("final_analysis", END)
    
    return workflow.compile()

graph = create_price_comparison_graph()