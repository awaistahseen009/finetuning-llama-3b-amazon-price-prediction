import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agents.graph import graph

load_dotenv(override=True)

app = FastAPI(title="Price Comparison API", version="1.0.0")

class PriceRequest(BaseModel):
    content: str

class PriceResponse(BaseModel):
    description: str
    predicted_price: str
    market_price: str
    market_analysis: dict
    sources: list
    comparison: dict

@app.post("/compare-price", response_model=PriceResponse)
async def compare_price(request: PriceRequest):
    try:
        print(f"Received request: {request.content}")
        
        initial_state = {
            "messages": [],
            "user_input": request.content,
            "predicted_price": None,
            "search_results": [],
            "scraped_data": [],
            "final_answer": None
        }
        
        print("Starting graph execution...")
        result = graph.invoke(initial_state)
        print(f"Graph execution completed: {result.get('final_answer') is not None}")
        
        final_answer = result["final_answer"]
        
        if not final_answer:
            print("No final answer generated")
            raise HTTPException(status_code=500, detail="Failed to generate analysis")
        
        print("Returning successful response")
        return PriceResponse(**final_answer)
    
    except Exception as e:
        print(f"Error in compare_price: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)