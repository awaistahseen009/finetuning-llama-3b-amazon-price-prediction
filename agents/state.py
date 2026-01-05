from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class PriceComparisonState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    predicted_price: Optional[float]
    search_results: List[dict]
    scraped_data: List[dict]
    final_answer: Optional[dict]