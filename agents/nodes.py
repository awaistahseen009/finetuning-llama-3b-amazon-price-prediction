import re
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from .state import PriceComparisonState
from .tools import predict_price_tool, search_web_tool, scrape_page_tool

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def price_prediction_node(state: PriceComparisonState) -> Dict[str, Any]:
    """Node to predict price using Modal API"""
    user_input = state["user_input"]
    try:
        predicted_price = predict_price_tool.invoke(user_input)
        return {
            "predicted_price": predicted_price,
            "messages": [AIMessage(content=f"Predicted price: ${predicted_price:.2f}")]
        }
    except Exception as e:
        return {
            "predicted_price": None,
            "messages": [AIMessage(content=str(e))]
        }

def web_search_node(state: PriceComparisonState) -> Dict[str, Any]:
    """Node to search web for product prices"""
    user_input = state["user_input"]
    search_query = f"{user_input} price buy online"
    
    search_results = search_web_tool.invoke(search_query)
    
    return {
        "search_results": search_results,
        "messages": [AIMessage(content=f"Found {len(search_results)} search results")]
    }

def scraping_node(state: PriceComparisonState) -> Dict[str, Any]:
    """Node to scrape pages from search results"""
    search_results = state.get("search_results", [])
    scraped_data = []
    
    for i, result in enumerate(search_results[:5]):
        if isinstance(result, dict) and "link" in result:
            url = result["link"]
            scraped_content = scrape_page_tool.invoke(url)
            scraped_data.append({
                "url": url,
                "title": result.get("title", ""),
                "content": scraped_content
            })
        elif isinstance(result, dict) and "url" in result:
            url = result["url"]
            scraped_content = scrape_page_tool.invoke(url)
            scraped_data.append({
                "url": url,
                "title": result.get("title", ""),
                "content": scraped_content
            })
    
    return {
        "scraped_data": scraped_data,
        "messages": [AIMessage(content=f"Scraped {len(scraped_data)} pages")]
    }

def extract_prices_from_content(content: str) -> List[float]:
    """Extract price values from scraped content"""
    price_patterns = [
        r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?) USD',
        r'Price[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?) dollars?'
    ]
    
    prices = []
    for pattern in price_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            try:
                price = float(match.replace(',', ''))
                if 1 <= price <= 10000:
                    prices.append(price)
            except ValueError:
                continue
    
    return list(set(prices))

def final_analysis_node(state: PriceComparisonState) -> Dict[str, Any]:
    """Node to analyze results and provide final answer"""
    predicted_price = state.get("predicted_price")
    scraped_data = state.get("scraped_data", [])
    user_input = state["user_input"]
    
    sources = []
    all_prices = []
    
    for data in scraped_data:
        content = data.get("content", {})
        if isinstance(content, dict):
            text_content = content.get("content", "")
        else:
            text_content = str(content)
        
        prices = extract_prices_from_content(text_content)
        if prices:
            sources.append({
                "url": data["url"],
                "title": data["title"],
                "prices": prices
            })
            all_prices.extend(prices)
    
    avg_market_price = sum(all_prices) / len(all_prices) if all_prices else None
    min_market_price = min(all_prices) if all_prices else None
    max_market_price = max(all_prices) if all_prices else None
    
    if predicted_price is None:
        prediction_status = "Price API not working"
        comparison = None
    else:
        prediction_status = f"${predicted_price:.2f}"
        comparison = {
            "prediction_vs_market": (predicted_price - avg_market_price) if avg_market_price else None,
            "accuracy_assessment": "within_range" if avg_market_price and abs(predicted_price - avg_market_price) < 50 else "outside_range"
        }
    
    # Generate summary description
    if all_prices:
        price_range = f"${min_market_price:.2f} to ${max_market_price:.2f}"
        avg_price_text = f"${avg_market_price:.2f}"
        summary = f"Based on {len(sources)} sources found online, the market price for '{user_input}' ranges from {price_range} with an average price of {avg_price_text}. "
        
        if predicted_price is not None:
            if avg_market_price:
                diff = abs(predicted_price - avg_market_price)
                if diff < 50:
                    summary += f"Our AI prediction of ${predicted_price:.2f} is very close to the market average, within ${diff:.2f}."
                else:
                    summary += f"Our AI prediction of ${predicted_price:.2f} differs from the market average by ${diff:.2f}."
            else:
                summary += f"Our AI prediction is ${predicted_price:.2f}."
        else:
            summary += "Our AI prediction service is currently unavailable."
    else:
        summary = f"No market pricing data was found online for '{user_input}'. "
        if predicted_price is not None:
            summary += f"However, our AI prediction estimates the price at ${predicted_price:.2f}."
        else:
            summary += "Our AI prediction service is also currently unavailable."
    
    final_answer = {
        "description": summary,
        "predicted_price": prediction_status,
        "market_price": avg_price_text if all_prices else "No market data found",
        "market_analysis": {
            "average_price": avg_market_price,
            "min_price": min_market_price,
            "max_price": max_market_price,
            "total_sources": len(sources)
        },
        "sources": sources,
        "comparison": comparison
    }
    
    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=f"Analysis complete. Found {len(sources)} sources with pricing data.")]
    }