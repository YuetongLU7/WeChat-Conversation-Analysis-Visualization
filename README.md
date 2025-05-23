# WeChat Sentiment Analysis

A web application for analyzing WeChat chat records and generating sentiment analysis reports.

## Features

- Upload and process WeChat chat records in CSV format
- Analyze chat patterns over time
- Generate word clouds for both participants
- Perform sentiment analysis on chat content
- Generate downloadable PDF reports

## Project Structure

The project consists of two main components:

### Frontend

- Built with Vue.js and Element UI
- Provides an intuitive interface for uploading chat records and viewing analysis results

### Backend

- Built with Flask (Python)
- Processes chat data and performs sentiment analysis
- Generates visualizations and reports

## Setup and Installation

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd wechat_sentiment_analysis/backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```
   python app.py
   ```
   The server will run on http://localhost:8082

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd wechat_sentiment_analysis/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run serve
   ```
   The application will be available at http://localhost:8083

## Usage

1. Start both the backend and frontend servers
2. Open your browser and navigate to http://localhost:8083
3. Upload a CSV file containing WeChat chat records
4. View the analysis results and download the generated report

## API Endpoints

- `POST /api/upload`: Upload and process chat records
- `GET /api/reports/<filename>`: Download generated reports
- `GET /api/test`: Test endpoint to verify backend connectivity

## License

MIT 