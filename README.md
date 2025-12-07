# Data Visualization Learning Platform

A Flask-based web application for learning data visualization techniques across four different domains: E-Commerce, Health, Education, and Finance.

## Features

- **4 Different Domains**: Each with realistic, domain-specific datasets
- **Multiple Visualization Libraries**: Matplotlib, Seaborn, Bokeh, and Plotly
- **Educational Content**: Each visualization includes detailed lesson descriptions explaining what, why, and how
- **Interactive Dashboards**: Some visualizations are interactive for better exploration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5003
```

## Navigation

- **Home Page**: Overview of all domains
- **Domain Pages**: Browse visualizations for each domain (E-Commerce, Health, Education, Finance)
- **Visualization Pages**: View individual visualizations with detailed lesson descriptions
- **Library Lessons**: Learn about each visualization library
  - `/lessons` - Overview of all libraries
  - `/lesson/matplotlib` - Matplotlib lesson
  - `/lesson/seaborn` - Seaborn lesson
  - `/lesson/bokeh` - Bokeh lesson
  - `/lesson/plotly` - Plotly lesson

## Domains and Visualizations

### 🛒 E-Commerce
- Daily Revenue Trend (Line Plot - Matplotlib)
- Category Sales Comparison (Bar Plot - Seaborn)
- Order Value Distribution (Histogram + KDE - Seaborn)
- Country-Device Heatmap (Heatmap - Seaborn)
- Interactive Revenue Dashboard (Plotly)

### 🧬 Health
- BMI Distribution (Histogram + KDE - Seaborn)
- Diagnosis-Based BMI Analysis (Box Plot - Seaborn)
- BMI vs Blood Pressure (Scatter Plot - Seaborn)
- Health Metrics Correlation (Correlation Heatmap - Seaborn)
- Interactive Patient Dashboard (Plotly)

### 📚 Education
- Subject Average Scores (Bar Plot - Seaborn)
- Homework vs Exam Performance (Scatter Plot - Seaborn)
- Program-Based Score Analysis (Box Plot - Seaborn)
- Grade Level Performance Trend (Line Plot - Matplotlib)
- Interactive Student Dashboard (Bokeh)

### 💰 Finance
- Interest Rate Distribution (Histogram + KDE - Seaborn)
- Risk Analysis Scatter (Scatter Plot - Seaborn)
- Bank Interest Rate Comparison (Box Plot - Seaborn)
- Financial Metrics Correlation (Correlation Heatmap - Seaborn)
- Interactive Loan Dashboard (Plotly)

## Project Structure

```
lesson demo/
├── app.py              # Flask application
├── data.py             # Dataset definitions
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── index.html
│   ├── domain.html
│   └── visualization.html
└── static/
    └── css/
        └── style.css   # Styling
```

## Technologies Used

- **Flask**: Web framework
- **Pandas**: Data manipulation
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations
- **Bokeh**: Interactive web visualizations
- **Plotly**: Interactive dashboards

