# Time-Series Data Processing Pipeline with XGBoost Forecasting  

A **generic time-series data processing pipeline** with a **React frontend**, a **Flask backend**, and **XGBoost for forecasting**. The backend handles **data retrieval, cleaning (missing values, duplicates, time zone inconsistencies), forecasting**, and storage in a CSV file. The frontend allows users to send time-series data, retrieve cleaned data, and get forecasts as visualized plots.  

## Features  

- **Data Ingestion**: Accepts time-series data via an API.  
- **Data Cleaning**: Handles missing values, removes duplicates, and ensures time zone consistency.  
- **Forecasting**: Uses **XGBoost** to predict future values based on historical data.  
- **Visualization**: Returns the forecast as an image/plot instead of raw JSON.  
- **Data Storage**: Saves processed data in a structured CSV file.  
- **Frontend Interface**: Built with React to interact with the API.  
- **Backend Processing**: Flask API for data handling, cleaning, and forecasting.  
- **Scalability**: Designed for general time-series applications, not limited to stocks.  

## Tech Stack  

- **Frontend**: React.js  
- **Backend**: Flask (Python)  
- **Machine Learning**: XGBoost  
- **Database**: CSV (for now, can be extended to PostgreSQL, MySQL, etc.)  
- **Libraries**:  
  - `pandas` for data processing  
  - `Flask` for API  
  - `XGBoost` for forecasting  
  - `Matplotlib`/`Seaborn` for visualization  
  - `Scikit-learn` for preprocessing  
  - `Axios` for frontend-backend communication  
  - `React Hooks` for state management  

## Installation  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.9+  
- Node.js (v16+)  
- npm/yarn  
- Virtualenv (optional but recommended)  

### Backend Setup (Flask + XGBoost)  

```sh
# Clone the repository
git clone https://github.com/yourusername/ForecastX.git  
cd ForecastX  

# Create a virtual environment (optional but recommended)
python -m venv venv  
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r backend/requirements.txt  

# Run the Flask server
cd backend  
python app.py  
```

### Frontend Setup (React)  

```sh
cd frontend  
npm install  # or yarn install  

# Start the React development server
npm start  # or yarn start  
```

## API Endpoints  

| Method | Endpoint  | Description                     |
|--------|----------|---------------------------------|
| **POST** | `/upload` | Uploads raw time-series data |
| **GET**  | `/fetch`  | Retrieves cleaned time-series data |
| **POST** | `/forecast` | Returns an **image (plot)** of the XGBoost-based forecast |

## How Forecasting Works  

1. The backend **preprocesses** the uploaded time-series data (handles missing values, formats timestamps, etc.).  
2. The **XGBoost** model is trained on historical data to capture trends and patterns.  
3. When a forecast request is made, the model **predicts future values** for a specified time range.  
4. The forecast is **visualized using Matplotlib/Seaborn** and returned as an **image (plot)** instead of raw JSON.  
5. The frontend displays the forecast image for the user.  

## Future Enhancements  

- Support for database storage (PostgreSQL, MySQL)  
- More advanced data transformations  
- User authentication  
- Interactive data visualization in the frontend  
- Option to choose different forecasting models (ARIMA, LSTM, Prophet)  

## Contributing  

1. Fork the repository  
2. Create a new branch (`git checkout -b feature-branch`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push to the branch (`git push origin feature-branch`)  
5. Open a Pull Request  
