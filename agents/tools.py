import os
from dotenv import load_dotenv
import requests
from typing import List, Dict, Any
from langchain_core.tools import tool

load_dotenv(override=True)

@tool
def predict_price_tool(content: str) -> float:
    """Predict price using the Modal API endpoint"""
    modal_url = os.getenv("MODAL_URL")
    try:
        print(f"Calling Modal API: {modal_url}/predict")
        print(f"Request payload: {{'content': '{content}'}}")
        
        response = requests.post(
            f"{modal_url}/predict",
            json={"content": content},
            timeout=120
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        
        if "price" in result:
            price = float(result["price"])
            print(f"Extracted price: {price}")
            return price
        else:
            print(f"No 'price' key in response: {result}")
            raise Exception("Price API not working - no price in response")
            
    except requests.exceptions.Timeout:
        print("Request timed out after 120 seconds")
        raise Exception("Price API not working - request timed out")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        raise Exception("Price API not working - request failed")
    except (ValueError, KeyError) as e:
        print(f"Response parsing error: {e}")
        raise Exception("Price API not working - invalid response format")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise Exception("Price API not working")

@tool
def search_web_tool(query: str) -> List[Dict[str, Any]]:
    """Search web for product prices using Serper API"""
    try:
        serper_api_key = os.getenv("SERPER_API_KEY")
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": 5
        }
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "organic" in data:
            for item in data["organic"][:5]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
        return results
    except Exception as e:
        return []

@tool
def scrape_page_tool(url: str) -> Dict[str, Any]:
    """Scrape a webpage for price information using FireCrawl"""
    try:
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        firecrawl_url = "https://api.firecrawl.dev/v0/scrape"
        payload = {
            "url": url,
            "pageOptions": {
                "onlyMainContent": True
            }
        }
        headers = {
            "Authorization": f"Bearer {firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(firecrawl_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return {
                "content": data.get("data", {}).get("content", ""),
                "title": data.get("data", {}).get("title", ""),
                "url": url
            }
        else:
            return {"error": "Failed to scrape", "content": "", "url": url}
    except Exception as e:
        return {"error": str(e), "content": "", "url": url}