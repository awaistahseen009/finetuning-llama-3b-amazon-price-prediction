# LLaMA-3 Price Prediction System

A comprehensive machine learning system that fine-tunes LLaMA-3.2-3B to predict product prices based on descriptions, with comparative analysis against traditional ML and deep learning approaches.

## Project Overview

This project implements an end-to-end price prediction system using multiple approaches:
- **Fine-tuned LLaMA-3.2-3B** (Primary approach)
- **Traditional Machine Learning** (Linear Regression, SVR, Tree-based methods)
- **Deep Learning** (MLP, RNN architectures)
- **Baseline Models** for comparison

The system achieves a **$40 average error**, outperforming all traditional approaches, and is deployed on Modal with A10G GPU for real-time inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LLaMA-3 Price Prediction System                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Data Pipeline  │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • Amazon Reviews│───▶│ • Data Curation │───▶│ • Traditional ML│
│ • Product Meta  │    │ • Text Cleaning │    │ • Deep Learning │
│ • Descriptions  │    │ • Feature Eng.  │    │ • LLM Fine-tune │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Query Rewriting │    │   Evaluation    │    │   Deployment    │
│                 │    │                 │    │                 │
│ • GPT-4o System │    │ • Error Analysis│    │ • Modal Cloud   │
│ • Prompt Format │    │ • Visualization │    │ • A10G GPU      │
│ • Text Cleanup  │    │ • Metrics (MSE) │    │ • FastAPI       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Agent System   │    │  Web Interface  │    │  Price Analysis │
│                 │    │                 │    │                 │
│ • LangGraph     │    │ • React + Vite  │    │ • AI Prediction │
│ • Web Search    │    │ • Tailwind CSS  │    │ • Market Data   │
│ • Web Scraping  │    │ • Real-time UI  │    │ • Comparison    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
├── fine-tuning-modules/          # Production ML pipeline
│   ├── config.py                 # Configuration management
│   ├── config.yaml               # Training hyperparameters
│   ├── model.py                  # Model & tokenizer setup
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Model evaluation
│   ├── inference.py              # Inference logic
│   ├── modal_app.py              # Modal deployment
│   ├── eval_config.py            # Evaluation configuration
│   ├── eval_config.yaml          # Evaluation parameters
│   └── eval_model.py             # Model loading for eval
│
├── notebooks/                    # Research & experimentation
│   ├── data_curation.ipynb       # Data collection & cleaning
│   ├── data_preparation.ipynb    # Feature engineering
│   ├── finetuning/               # LLM fine-tuning experiments
│   │   ├── finetuning-llama-3b.ipynb
│   │   ├── finetuning-evaluate.ipynb
│   │   ├── prompt_prep.ipynb
│   │   └── util.py               # Evaluation utilities
│   └── ml_experiments/           # Traditional ML & DL experiments
│       ├── traditional_ml/
│       │   ├── linear_regression/
│       │   ├── SVM/
│       │   ├── tree_based_methods/
│       │   └── mean_estimation/
│       └── deep_learning/
│           ├── MLP/
│           └── minbpe/           # Tokenization experiments
│
├── pricer/                       # Data processing & evaluation
│   ├── items.py                  # Item data model
│   ├── loader.py                 # Dataset loading
│   ├── parser.py                 # Data parsing & cleaning
│   ├── batch_runner.py           # Batch processing
│   └── evaluate.py               # Evaluation framework
│
├── agents/                       # LangGraph agent system
│   ├── __init__.py               # Package initialization
│   ├── state.py                  # Agent state management
│   ├── tools.py                  # API tools (Modal, Serper, FireCrawl)
│   ├── nodes.py                  # Graph nodes and logic
│   └── graph.py                  # LangGraph workflow
│
├── frontend/                     # React web interface
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   ├── main.jsx              # Application entry point
│   │   └── index.css             # Tailwind CSS styles
│   ├── public/                   # Static assets
│   ├── index.html                # HTML template
│   ├── package.json              # Frontend dependencies
│   ├── vite.config.js            # Vite configuration
│   └── tailwind.config.js        # Tailwind configuration
│
├── jsonl/                        # Data files
│   ├── 0_1000.jsonl             # Sample batch data
│   └── batch_results.jsonl       # Batch processing results
│
├── app.py                        # FastAPI backend server
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## Installation & Setup

### Prerequisites

**Backend Requirements:**
- Python 3.10 or higher
- pip package manager

**Frontend Requirements:**
- Node.js 16 or higher
- npm or yarn package manager

### Backend Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd finetuning-llama-3b-amazon-price-prediction
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Copy the template and fill in your API keys
cp readme.env .env
# Edit .env with your actual API keys:
# - OPENAI_API_KEY
# - FIRECRAWL_API_KEY  
# - SERPER_API_KEY
# - HF_TOKEN (Hugging Face)
# - GROQ_API_KEY
# - WANDB_API_KEY
```

### Frontend Installation

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

## Running the Application

### Start the Backend Server

1. **Navigate to project root:**
```bash
cd /path/to/finetuning-llama-3b-amazon-price-prediction
```

2. **Activate virtual environment:**
```bash
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Start the FastAPI server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Start the Frontend Development Server

1. **Open a new terminal and navigate to frontend:**
```bash
cd frontend
```

2. **Start the Vite development server:**
```bash
npm run dev
```

The frontend will be available at: `http://localhost:3000`

### Production Build (Frontend)

```bash
cd frontend
npm run build
npm run preview
```

## Frontend Features

### User Interface:
- **Modern Design** - Gradient backgrounds and elegant card layouts
- **Responsive Layout** - Works perfectly on desktop, tablet, and mobile
- **Real-time Feedback** - Loading states with progress indicators
- **Interactive Elements** - Expandable sections and hover effects

### Key Components:
1. **Price Input Form** - Large textarea for product descriptions
2. **AI Prediction Card** - Shows ML model prediction in blue
3. **Market Average Card** - Displays web-scraped market data in green
4. **Market Analysis Grid** - Min/max/average prices and source count
5. **Sources Dropdown** - Expandable list of scraped websites with prices
6. **Accuracy Assessment** - Color-coded prediction accuracy indicators

### User Experience:
- **Progress Tracking** - Shows "Getting AI prediction...", "Searching the web...", etc.
- **Error Handling** - User-friendly error messages and timeout handling
- **External Links** - Direct links to source websites
- **Price Visualization** - Color-coded price tags and comparison metrics

## API Usage

### Price Comparison Endpoint

```bash
curl -X POST "http://localhost:8000/compare-price" \
  -H "Content-Type: application/json" \
  -d '{"content": "Used MacBook Pro 14-inch M1 Pro, 16GB RAM, great condition"}'
```

### Response Format

```json
{
  "description": "Based on 3 sources found online, the market price for 'Used MacBook Pro...' ranges from $1,200.00 to $1,800.00 with an average price of $1,500.00. Our AI prediction of $1,450.00 is very close to the market average, within $50.00.",
  "predicted_price": "$1,450.00",
  "market_price": "$1,500.00",
  "market_analysis": {
    "average_price": 1500.0,
    "min_price": 1200.0,
    "max_price": 1800.0,
    "total_sources": 3
  },
  "sources": [
    {
      "url": "https://example.com/macbook",
      "title": "MacBook Pro for Sale",
      "prices": [1450.0, 1500.0]
    }
  ],
  "comparison": {
    "prediction_vs_market": -50.0,
    "accuracy_assessment": "within_range"
  }
}
```

## System Workflow

### Agent Pipeline:
1. **Price Prediction** - Calls Modal API for AI prediction
2. **Web Search** - Uses Serper API to find product listings (max 5 results)
3. **Web Scraping** - Uses FireCrawl API to extract price data (1 page per result)
4. **Analysis** - Compares AI prediction with market data
5. **Response** - Returns comprehensive price analysis

### Conditional Flow:
- If AI prediction fails → Skip to web search only
- If web search fails → Return AI prediction only  
- If both fail → Return appropriate error messages

## Performance Results

| Approach | Average Error | MSE | R² Score | Notes |
|----------|---------------|-----|----------|-------|
| **LLaMA-3 Fine-tuned** | **$40** | **Low** | **High** | **Best performer** |
| Traditional ML | $60-80 | Medium | Medium | Baseline methods |
| Deep Learning | $50-70 | Medium | Medium | Neural networks |
| Mean Estimation | $100+ | High | Low | Simple baseline |

### Key Achievements
- 40% better than traditional ML approaches
- Real-time inference on Modal cloud
- Scalable architecture with GPU optimization
- Comprehensive evaluation framework
- Beautiful web interface with real-time market data
- Agent-based system with web search and scraping

## Development

### Backend Development:
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --reload

# Run tests (if available)
python -m pytest
```

### Frontend Development:
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Environment Variables

Required environment variables (see `readme.env` for template):

```bash
# OpenAI API for LLM operations
OPENAI_API_KEY=your_openai_api_key_here

# Modal deployment URL (pre-configured)
MODAL_URL=https://awaistahseen009--llama-price-predictor-1-fastapi-app.modal.run

# FireCrawl API for web scraping
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Serper API for web search
SERPER_API_KEY=your_serper_api_key_here

# Additional ML/Training APIs
HF_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
```

## Future Enhancements

- Multi-modal inputs: Image + text processing
- Dynamic pricing: Market trend integration
- Category-specific models: Specialized fine-tuning
- Uncertainty quantification: Confidence intervals
- A/B testing framework: Model comparison in production
- Mobile app: React Native implementation
- Real-time updates: WebSocket integration
- User accounts: Save search history and favorites

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Meta AI** for LLaMA-3.2-3B model
- **Hugging Face** for transformers and datasets
- **Modal** for serverless GPU deployment
- **Amazon** for the product reviews dataset
- **LangChain** for agent framework
- **Vite & React** for modern frontend development

---

**Built for accurate price prediction using state-of-the-art LLM fine-tuning and intelligent web agents**