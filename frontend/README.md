# Price Predictor Frontend

A beautiful React application for the Price Predictor system built with Vite and Tailwind CSS.

## Features

- 🎨 Beautiful, modern UI with gradient backgrounds
- 📱 Fully responsive design
- 🔍 Real-time price prediction and market analysis
- 📊 Interactive price comparison charts
- 🔗 Expandable sources dropdown with external links
- ⚡ Fast development with Vite
- 🎯 Built with React 18 and Tailwind CSS

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and visit `http://localhost:3000`

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## API Integration

The frontend is configured to proxy API requests to the backend running on `http://localhost:8000`. Make sure your backend server is running before using the application.

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # Application entry point
│   └── index.css        # Global styles and Tailwind imports
├── public/              # Static assets
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── tailwind.config.js   # Tailwind CSS configuration
```

## Technologies Used

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client for API requests
- **Lucide React** - Beautiful icons
- **PostCSS** - CSS processing

## Features Overview

### Price Input
- Large, user-friendly textarea for product descriptions
- Real-time validation and loading states
- Elegant form design with focus states

### Results Display
- **AI Prediction Card** - Shows the predicted price from your ML model
- **Market Average Card** - Displays the average market price from web scraping
- **Market Analysis Grid** - Min, max, average prices and source count
- **Sources Dropdown** - Expandable list of scraped websites with prices
- **Accuracy Assessment** - Shows how close the prediction is to market data

### UI/UX Features
- Gradient backgrounds and card shadows
- Smooth hover animations and transitions
- Loading spinners and error handling
- Responsive design for all screen sizes
- Color-coded price tags and accuracy indicators