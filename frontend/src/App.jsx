import React, { useState } from 'react'
import axios from 'axios'
import { Search, DollarSign, TrendingUp, ExternalLink, ChevronDown, ChevronUp, Loader2, AlertCircle } from 'lucide-react'

function App() {
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [showSources, setShowSources] = useState(false)
  const [loadingStep, setLoadingStep] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!description.trim()) return

    setLoading(true)
    setError('')
    setResult(null)
    setLoadingStep('Getting AI prediction...')

    try {
      // Simulate progress updates
      const progressTimer = setInterval(() => {
        setLoadingStep(prev => {
          if (prev === 'Getting AI prediction...') return 'Searching the web...'
          if (prev === 'Searching the web...') return 'Scraping price data...'
          if (prev === 'Scraping price data...') return 'Analyzing results...'
          return 'Almost done...'
        })
      }, 15000)

      // Increase timeout to 3 minutes for the frontend request
      const response = await axios.post('/api/compare-price', {
        content: description
      }, {
        timeout: 180000 // 3 minutes
      })
      
      clearInterval(progressTimer)
      setResult(response.data)
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The analysis is taking longer than expected. Please try again.')
      } else {
        setError(err.response?.data?.detail || 'Failed to get price prediction')
      }
    } finally {
      setLoading(false)
      setLoadingStep('')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <DollarSign className="w-12 h-12 text-primary-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-800">Price Predictor</h1>
          </div>
          <p className="text-gray-600 text-lg">
            Get AI-powered price predictions and real-time market analysis
          </p>
        </div>

        {/* Input Form */}
        <div className="bg-white rounded-2xl card-shadow p-8 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                Paste the description of the item
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter product description (e.g., Used MacBook Pro 14-inch M1 Pro, 16GB RAM, great condition)"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg input-focus resize-none"
                rows={4}
                required
              />
            </div>
            
            <button
              type="submit"
              disabled={loading || !description.trim()}
              className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg button-hover disabled:transform-none disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <div className="flex flex-col items-center">
                  <div className="flex items-center mb-2">
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Analyzing...
                  </div>
                  <p className="text-sm text-gray-600">{loadingStep}</p>
                  <p className="text-xs text-gray-500 mt-1">This may take up to 2 minutes</p>
                </div>
              ) : (
                <>
                  <Search className="w-5 h-5 mr-2" />
                  Get Price Prediction
                </>
              )}
            </button>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8 flex items-center">
            <AlertCircle className="w-5 h-5 text-red-500 mr-3" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="bg-white rounded-2xl card-shadow p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                <TrendingUp className="w-6 h-6 mr-3 text-primary-600" />
                Price Analysis
              </h2>
              <p className="text-gray-700 leading-relaxed mb-6">
                {result.description}
              </p>
              
              {/* Price Cards */}
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl p-6 text-white">
                  <h3 className="text-lg font-semibold mb-2">AI Prediction</h3>
                  <p className="text-3xl font-bold">{result.predicted_price}</p>
                </div>
                
                <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-xl p-6 text-white">
                  <h3 className="text-lg font-semibold mb-2">Market Average</h3>
                  <p className="text-3xl font-bold">{result.market_price}</p>
                </div>
              </div>
            </div>

            {/* Market Analysis */}
            <div className="bg-white rounded-2xl card-shadow p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Market Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Min Price</p>
                  <p className="text-lg font-semibold text-gray-800">
                    {result.market_analysis.min_price ? `$${result.market_analysis.min_price.toFixed(2)}` : 'N/A'}
                  </p>
                </div>
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Max Price</p>
                  <p className="text-lg font-semibold text-gray-800">
                    {result.market_analysis.max_price ? `$${result.market_analysis.max_price.toFixed(2)}` : 'N/A'}
                  </p>
                </div>
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Average</p>
                  <p className="text-lg font-semibold text-gray-800">
                    {result.market_analysis.average_price ? `$${result.market_analysis.average_price.toFixed(2)}` : 'N/A'}
                  </p>
                </div>
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Sources</p>
                  <p className="text-lg font-semibold text-gray-800">
                    {result.market_analysis.total_sources}
                  </p>
                </div>
              </div>
            </div>

            {/* Sources Dropdown */}
            {result.sources && result.sources.length > 0 && (
              <div className="bg-white rounded-2xl card-shadow p-8">
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="w-full flex items-center justify-between text-xl font-bold text-gray-800 hover:text-primary-600 transition-colors"
                >
                  <span>Price Sources ({result.sources.length})</span>
                  {showSources ? (
                    <ChevronUp className="w-6 h-6" />
                  ) : (
                    <ChevronDown className="w-6 h-6" />
                  )}
                </button>
                
                {showSources && (
                  <div className="mt-6 space-y-4">
                    {result.sources.map((source, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-800 mb-2">
                              {source.title || 'Untitled Source'}
                            </h4>
                            <div className="flex flex-wrap gap-2 mb-2">
                              {source.prices.map((price, priceIndex) => (
                                <span
                                  key={priceIndex}
                                  className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm font-medium"
                                >
                                  ${price.toFixed(2)}
                                </span>
                              ))}
                            </div>
                          </div>
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="ml-4 text-primary-600 hover:text-primary-700 transition-colors"
                          >
                            <ExternalLink className="w-5 h-5" />
                          </a>
                        </div>
                        <p className="text-sm text-gray-500 truncate">
                          {source.url}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Comparison */}
            {result.comparison && (
              <div className="bg-white rounded-2xl card-shadow p-8">
                <h3 className="text-xl font-bold text-gray-800 mb-4">Prediction Accuracy</h3>
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-gray-700">Assessment:</span>
                  <span className={`font-semibold px-3 py-1 rounded-full text-sm ${
                    result.comparison.accuracy_assessment === 'within_range' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {result.comparison.accuracy_assessment === 'within_range' ? 'Accurate' : 'Outside Range'}
                  </span>
                </div>
                {result.comparison.prediction_vs_market && (
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <p className="text-blue-800">
                      <strong>Difference:</strong> ${Math.abs(result.comparison.prediction_vs_market).toFixed(2)} 
                      {result.comparison.prediction_vs_market > 0 ? ' above' : ' below'} market average
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App