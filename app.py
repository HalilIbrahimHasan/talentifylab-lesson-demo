from flask import Flask, render_template, send_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import numpy as np

from data import amazon_data, health_data, education_data, finance_data
import os

# Load student study success data
student_data_path = os.path.join(os.path.dirname(__file__), 'student_study_success.csv')
student_data = pd.read_csv(student_data_path)
student_data = student_data.dropna()  # Remove empty rows

# Load home loan applicants data
loan_data_path = os.path.join(os.path.dirname(__file__), 'home_loan_applicants.csv')
loan_data = pd.read_csv(loan_data_path)
loan_data = loan_data.dropna()  # Remove empty rows

# Load e-commerce 50 records data
ecommerce_csv_path = os.path.join(os.path.dirname(__file__), 'ecommerce_50_records.csv')
ecommerce_csv_data = pd.read_csv(ecommerce_csv_path)
ecommerce_csv_data = ecommerce_csv_data.dropna()  # Remove empty rows
ecommerce_csv_data['order_date'] = pd.to_datetime(ecommerce_csv_data['order_date'])
ecommerce_csv_data['revenue'] = ecommerce_csv_data['price'] * ecommerce_csv_data['quantity']

# Load health clinic data
health_clinic_path = os.path.join(os.path.dirname(__file__), 'health_data_clinic.csv')
health_clinic_data = pd.read_csv(health_clinic_path)
health_clinic_data = health_clinic_data.dropna()  # Remove empty rows

# Load interactive 3D dataset
interactive_3d_path = os.path.join(os.path.dirname(__file__), 'interactive_3d_dataset.csv')
interactive_3d_data = pd.read_csv(interactive_3d_path)
interactive_3d_data = interactive_3d_data.dropna()  # Remove empty rows

app = Flask(__name__)
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ecommerce')
def ecommerce():
    return render_template('domain.html', 
                         domain='E-Commerce',
                         domain_id='ecommerce',
                         description='Amazon-style e-commerce sales data analysis',
                         datasets=[
                             {'name': 'Daily Revenue Trend', 'type': 'line', 'library': 'matplotlib'},
                             {'name': 'Category Sales Comparison', 'type': 'bar', 'library': 'seaborn'},
                             {'name': 'Order Value Distribution', 'type': 'histogram', 'library': 'seaborn'},
                             {'name': 'Country-Device Heatmap', 'type': 'heatmap', 'library': 'seaborn'},
                             {'name': 'Interactive Revenue Dashboard', 'type': 'interactive', 'library': 'plotly'}
                         ])

@app.route('/ecommerce/<viz_type>')
def ecommerce_viz(viz_type):
    if viz_type == 'line':
        return render_template('visualization.html',
                             domain='E-Commerce',
                             viz_name='Daily Revenue Trend',
                             viz_type='Line Plot',
                             library='Matplotlib',
                             description='This line plot visualizes daily total revenue trends over time. In e-commerce analytics, tracking revenue trends helps identify sales patterns, seasonal effects, and business growth. The x-axis shows order dates, and the y-axis displays total revenue (price × quantity) for each day.',
                             lesson='Line plots are essential for time series analysis in e-commerce. They help businesses understand sales performance over time, identify peak sales days, and forecast future revenue. This visualization uses the E-Commerce domain to demonstrate how to track business metrics chronologically.',
                             plot_data=generate_ecommerce_line())
    
    elif viz_type == 'bar':
        return render_template('visualization.html',
                             domain='E-Commerce',
                             viz_name='Category Sales Comparison',
                             viz_type='Bar Plot',
                             library='Seaborn',
                             description='This bar chart compares total sales across different product categories. Bar plots are ideal for comparing categorical data, making it easy to identify which product categories generate the most revenue. Each bar represents a category, and the height shows total sales value.',
                             lesson='Bar plots are fundamental in e-commerce analytics for comparing performance across categories, regions, or time periods. This visualization uses the E-Commerce domain to show how businesses can identify top-performing product categories and allocate resources accordingly.',
                             plot_data=generate_ecommerce_bar())
    
    elif viz_type == 'histogram':
        return render_template('visualization.html',
                             domain='E-Commerce',
                             viz_name='Order Value Distribution',
                             viz_type='Histogram with KDE',
                             library='Seaborn',
                             description='This histogram with Kernel Density Estimation (KDE) shows the distribution of order values (cart totals). It helps understand customer spending patterns, identify typical order sizes, and detect outliers. The KDE curve provides a smooth estimate of the probability distribution.',
                             lesson='Histograms with KDE are powerful tools for understanding data distributions in e-commerce. They reveal customer behavior patterns, such as typical spending amounts and whether the distribution is normal, skewed, or has multiple peaks. This visualization uses the E-Commerce domain to demonstrate distribution analysis.',
                             plot_data=generate_ecommerce_histogram())
    
    elif viz_type == 'heatmap':
        return render_template('visualization.html',
                             domain='E-Commerce',
                             viz_name='Country-Device Heatmap',
                             viz_type='Heatmap',
                             library='Seaborn',
                             description='This heatmap displays the relationship between countries and device types used for purchases. Heatmaps use color intensity to represent values, making it easy to spot patterns and correlations. Darker colors indicate higher sales volumes.',
                             lesson='Heatmaps are excellent for visualizing relationships between two categorical variables in e-commerce. They help identify market segments, understand customer preferences by region, and optimize marketing strategies. This visualization uses the E-Commerce domain to show cross-tabulation analysis.',
                             plot_data=generate_ecommerce_heatmap())
    
    elif viz_type == 'interactive':
        return render_template('visualization.html',
                             domain='E-Commerce',
                             viz_name='Interactive Revenue Dashboard',
                             viz_type='Interactive Plot',
                             library='Plotly',
                             description='This interactive plotly visualization allows users to explore revenue data dynamically. Interactive visualizations enable zooming, panning, and hovering to see detailed information, making them ideal for dashboards and presentations.',
                             lesson='Interactive visualizations enhance data exploration by allowing users to interact with the data. In e-commerce, interactive dashboards help stakeholders explore sales data, filter by categories, and drill down into specific time periods. This visualization uses the E-Commerce domain to demonstrate modern data visualization techniques.',
                             plot_data=generate_ecommerce_plotly())

@app.route('/health')
def health():
    return render_template('domain.html',
                         domain='Health',
                         domain_id='health',
                         description='Labcorp/BCBS-style healthcare patient data analysis',
                         datasets=[
                             {'name': 'BMI Distribution', 'type': 'histogram', 'library': 'seaborn'},
                             {'name': 'Diagnosis-Based BMI Analysis', 'type': 'boxplot', 'library': 'seaborn'},
                             {'name': 'BMI vs Blood Pressure', 'type': 'scatter', 'library': 'seaborn'},
                             {'name': 'Health Metrics Correlation', 'type': 'heatmap', 'library': 'seaborn'},
                             {'name': 'Interactive Patient Dashboard', 'type': 'interactive', 'library': 'plotly'}
                         ])

@app.route('/health/<viz_type>')
def health_viz(viz_type):
    if viz_type == 'histogram':
        return render_template('visualization.html',
                             domain='Health',
                             viz_name='BMI Distribution',
                             viz_type='Histogram with KDE',
                             library='Seaborn',
                             description='This histogram with KDE shows the distribution of Body Mass Index (BMI) values across patients. Understanding BMI distribution helps healthcare providers identify population health trends, assess obesity prevalence, and plan preventive care strategies.',
                             lesson='Histograms with KDE are crucial in healthcare analytics for understanding patient population characteristics. They help identify normal ranges, outliers, and distribution patterns that inform clinical decision-making. This visualization uses the Health domain to demonstrate population health analysis.',
                             plot_data=generate_health_histogram())
    
    elif viz_type == 'boxplot':
        return render_template('visualization.html',
                             domain='Health',
                             viz_name='Diagnosis-Based BMI Analysis',
                             viz_type='Box Plot',
                             library='Seaborn',
                             description='This box plot compares BMI values across different diagnoses. Box plots show quartiles, medians, and outliers, making it easy to compare distributions between groups. They reveal how BMI varies by medical condition.',
                             lesson='Box plots are essential in healthcare for comparing patient metrics across different diagnoses or treatment groups. They help identify which conditions are associated with higher or lower values, detect outliers, and understand variability. This visualization uses the Health domain to demonstrate comparative analysis in medical research.',
                             plot_data=generate_health_boxplot())
    
    elif viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='Health',
                             viz_name='BMI vs Blood Pressure',
                             viz_type='Scatter Plot',
                             library='Seaborn',
                             description='This scatter plot explores the relationship between BMI and systolic blood pressure, colored by diagnosis. Scatter plots reveal correlations between variables and help identify risk factors. Different colors represent different diagnoses.',
                             lesson='Scatter plots are fundamental in healthcare research for identifying relationships between health metrics. They help discover correlations, such as how BMI affects blood pressure, and inform treatment strategies. This visualization uses the Health domain to demonstrate correlation analysis in medical data.',
                             plot_data=generate_health_scatter())
    
    elif viz_type == 'heatmap':
        return render_template('visualization.html',
                             domain='Health',
                             viz_name='Health Metrics Correlation',
                             viz_type='Correlation Heatmap',
                             library='Seaborn',
                             description='This correlation heatmap shows relationships between numeric health metrics (age, BMI, blood pressure, lab results, outcome scores). Correlation values range from -1 to 1, with colors indicating strength and direction of relationships.',
                             lesson='Correlation heatmaps are powerful tools in healthcare analytics for understanding how different health metrics relate to each other. They help identify risk factors, predict outcomes, and guide treatment decisions. This visualization uses the Health domain to demonstrate multivariate analysis in medical research.',
                             plot_data=generate_health_heatmap())
    
    elif viz_type == 'interactive':
        return render_template('visualization.html',
                             domain='Health',
                             viz_name='Interactive Patient Dashboard',
                             viz_type='Interactive Plot',
                             library='Plotly',
                             description='This interactive plotly visualization allows exploration of patient data with filtering and hover details. Interactive dashboards enable healthcare professionals to explore patient populations dynamically.',
                             lesson='Interactive visualizations in healthcare enable dynamic exploration of patient data, helping clinicians and researchers identify patterns, outliers, and relationships. This visualization uses the Health domain to demonstrate modern healthcare analytics tools.',
                             plot_data=generate_health_plotly())

@app.route('/education')
def education():
    return render_template('domain.html',
                         domain='Education',
                         domain_id='education',
                         description='School exam scores and student performance analysis',
                         datasets=[
                             {'name': 'Subject Average Scores', 'type': 'bar', 'library': 'seaborn'},
                             {'name': 'Homework vs Exam Performance', 'type': 'scatter', 'library': 'seaborn'},
                             {'name': 'Program-Based Score Analysis', 'type': 'boxplot', 'library': 'seaborn'},
                             {'name': 'Grade Level Performance Trend', 'type': 'line', 'library': 'matplotlib'},
                             {'name': 'Interactive Student Dashboard', 'type': 'interactive', 'library': 'bokeh'}
                         ])

@app.route('/education/<viz_type>')
def education_viz(viz_type):
    if viz_type == 'bar':
        return render_template('visualization.html',
                             domain='Education',
                             viz_name='Subject Average Scores',
                             viz_type='Bar Plot',
                             library='Seaborn',
                             description='This bar chart compares average exam scores across different subjects. It helps educators identify which subjects students perform best in and where additional support may be needed. Each bar represents a subject, showing the mean exam score.',
                             lesson='Bar plots are essential in education analytics for comparing performance across subjects, classes, or programs. They help educators identify strengths and weaknesses in the curriculum and allocate teaching resources effectively. This visualization uses the Education domain to demonstrate academic performance analysis.',
                             plot_data=generate_education_bar())
    
    elif viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='Education',
                             viz_name='Homework vs Exam Performance',
                             viz_type='Scatter Plot',
                             library='Seaborn',
                             description='This scatter plot explores the relationship between homework hours and exam scores, colored by program type (Regular, Honors, Support). It helps identify if more homework correlates with better exam performance.',
                             lesson='Scatter plots in education help identify relationships between study habits and academic performance. They reveal whether factors like homework time, attendance, or program type correlate with success. This visualization uses the Education domain to demonstrate educational research methods.',
                             plot_data=generate_education_scatter())
    
    elif viz_type == 'boxplot':
        return render_template('visualization.html',
                             domain='Education',
                             viz_name='Program-Based Score Analysis',
                             viz_type='Box Plot',
                             library='Seaborn',
                             description='This box plot compares exam scores across different academic programs (Regular, Honors, Support). Box plots show score distributions, medians, and outliers, helping identify program effectiveness.',
                             lesson='Box plots are valuable in education for comparing performance across different programs, classes, or teaching methods. They help educators understand score distributions, identify outliers, and evaluate program effectiveness. This visualization uses the Education domain to demonstrate comparative educational analysis.',
                             plot_data=generate_education_boxplot())
    
    elif viz_type == 'line':
        return render_template('visualization.html',
                             domain='Education',
                             viz_name='Grade Level Performance Trend',
                             viz_type='Line Plot',
                             library='Matplotlib',
                             description='This line plot shows how average exam scores change across grade levels. It helps track academic progress over time and identify trends in student performance as they advance through school.',
                             lesson='Line plots in education help track trends over time, such as performance progression across grade levels or improvement over semesters. They are essential for longitudinal studies in education. This visualization uses the Education domain to demonstrate time-based academic analysis.',
                             plot_data=generate_education_line())
    
    elif viz_type == 'interactive':
        return render_template('visualization.html',
                             domain='Education',
                             viz_name='Interactive Student Dashboard',
                             viz_type='Interactive Plot',
                             library='Bokeh',
                             description='This interactive Bokeh visualization allows exploration of student performance data with tooltips and zooming capabilities. Interactive dashboards help educators and administrators explore student data dynamically.',
                             lesson='Interactive visualizations in education enable dynamic exploration of student performance data, helping educators identify patterns, outliers, and relationships. This visualization uses the Education domain to demonstrate modern educational analytics tools.',
                             plot_data=generate_education_bokeh())

@app.route('/finance')
def finance():
    return render_template('domain.html',
                         domain='Finance',
                         domain_id='finance',
                         description='Freddie Mac/Bank of America mortgage loan analysis',
                         datasets=[
                             {'name': 'Interest Rate Distribution', 'type': 'histogram', 'library': 'seaborn'},
                             {'name': 'Risk Analysis Scatter', 'type': 'scatter', 'library': 'seaborn'},
                             {'name': 'Bank Interest Rate Comparison', 'type': 'boxplot', 'library': 'seaborn'},
                             {'name': 'Financial Metrics Correlation', 'type': 'heatmap', 'library': 'seaborn'},
                             {'name': 'Interactive Loan Dashboard', 'type': 'interactive', 'library': 'plotly'}
                         ])

@app.route('/finance/<viz_type>')
def finance_viz(viz_type):
    if viz_type == 'histogram':
        return render_template('visualization.html',
                             domain='Finance',
                             viz_name='Interest Rate Distribution',
                             viz_type='Histogram with KDE',
                             library='Seaborn',
                             description='This histogram with KDE shows the distribution of mortgage interest rates. Understanding interest rate distributions helps financial institutions set competitive rates and assess market conditions.',
                             lesson='Histograms with KDE are essential in finance for understanding the distribution of key metrics like interest rates, loan amounts, and credit scores. They help identify typical values, outliers, and market trends. This visualization uses the Finance domain to demonstrate financial distribution analysis.',
                             plot_data=generate_finance_histogram())
    
    elif viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='Finance',
                             viz_name='Risk Analysis Scatter',
                             viz_type='Scatter Plot',
                             library='Seaborn',
                             description='This scatter plot analyzes the relationship between interest rates and debt-to-income (DTI) ratios, colored by default status. It helps identify risk factors and predict loan defaults.',
                             lesson='Scatter plots are crucial in finance for risk analysis, identifying relationships between variables like interest rates, DTI ratios, and credit scores. They help financial institutions assess loan risk and make informed lending decisions. This visualization uses the Finance domain to demonstrate financial risk analysis.',
                             plot_data=generate_finance_scatter())
    
    elif viz_type == 'boxplot':
        return render_template('visualization.html',
                             domain='Finance',
                             viz_name='Bank Interest Rate Comparison',
                             viz_type='Box Plot',
                             library='Seaborn',
                             description='This box plot compares interest rates offered by different banks. It helps borrowers compare lenders and helps banks understand competitive positioning in the market.',
                             lesson='Box plots in finance help compare metrics across different institutions, products, or time periods. They reveal distribution differences, outliers, and competitive positioning. This visualization uses the Finance domain to demonstrate comparative financial analysis.',
                             plot_data=generate_finance_boxplot())
    
    elif viz_type == 'heatmap':
        return render_template('visualization.html',
                             domain='Finance',
                             viz_name='Financial Metrics Correlation',
                             viz_type='Correlation Heatmap',
                             library='Seaborn',
                             description='This correlation heatmap shows relationships between financial metrics (interest rates, loan terms, principal, income, DTI ratio, credit score). Understanding these correlations helps assess risk and predict loan performance.',
                             lesson='Correlation heatmaps are powerful in finance for understanding relationships between financial variables. They help identify risk factors, predict defaults, and optimize lending strategies. This visualization uses the Finance domain to demonstrate multivariate financial analysis.',
                             plot_data=generate_finance_heatmap())
    
    elif viz_type == 'interactive':
        return render_template('visualization.html',
                             domain='Finance',
                             viz_name='Interactive Loan Dashboard',
                             viz_type='Interactive Plot',
                             library='Plotly',
                             description='This interactive plotly visualization allows exploration of loan data with filtering and detailed hover information. Interactive dashboards help financial analysts explore loan portfolios dynamically.',
                             lesson='Interactive visualizations in finance enable dynamic exploration of loan data, helping analysts identify patterns, assess risk, and make data-driven decisions. This visualization uses the Finance domain to demonstrate modern financial analytics tools.',
                             plot_data=generate_finance_plotly())

# E-Commerce Visualization Generators
def generate_ecommerce_line():
    daily_revenue = amazon_data.groupby('order_date')['revenue'].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(daily_revenue['order_date'], daily_revenue['revenue'], marker='o', linewidth=2, markersize=8)
    plt.title('Daily Revenue Trend - E-Commerce Sales', fontsize=16, fontweight='bold')
    plt.xlabel('Order Date', fontsize=12)
    plt.ylabel('Total Revenue ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_bar():
    category_sales = amazon_data.groupby('category')['revenue'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_sales.index, y=category_sales.values, palette='viridis')
    plt.title('Category Sales Comparison - Total Revenue by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Revenue ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(amazon_data['revenue'], kde=True, bins=10, color='skyblue', edgecolor='black')
    plt.title('Order Value Distribution - Cart Total Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Order Value ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_heatmap():
    pivot_data = amazon_data.groupby(['country', 'device']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Orders'})
    plt.title('Country-Device Sales Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Device Type', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_plotly():
    fig = px.scatter(amazon_data, x='order_date', y='revenue', 
                     color='category', size='quantity',
                     hover_data=['country', 'device', 'customer_segment'],
                     title='Interactive E-Commerce Revenue Dashboard')
    fig.update_layout(height=600)
    return json.dumps(fig, cls=PlotlyJSONEncoder)

# Health Visualization Generators
def generate_health_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(health_data['bmi'], kde=True, bins=10, color='lightcoral', edgecolor='black')
    plt.title('BMI Distribution - Patient Population Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Body Mass Index (BMI)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(health_data['bmi'].mean(), color='red', linestyle='--', label=f'Mean: {health_data["bmi"].mean():.1f}')
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_boxplot():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=health_data, x='diagnosis', y='bmi', palette='Set2')
    plt.title('BMI Analysis by Diagnosis', fontsize=16, fontweight='bold')
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Body Mass Index (BMI)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_scatter():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=health_data, x='bmi', y='systolic_bp', hue='diagnosis', 
                    s=100, alpha=0.7, palette='Set1')
    plt.title('BMI vs Systolic Blood Pressure by Diagnosis', fontsize=16, fontweight='bold')
    plt.xlabel('Body Mass Index (BMI)', fontsize=12)
    plt.ylabel('Systolic Blood Pressure (mmHg)', fontsize=12)
    plt.legend(title='Diagnosis')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_heatmap():
    numeric_cols = ['age', 'bmi', 'systolic_bp', 'lab_result', 'outcome_score']
    corr_matrix = health_data[numeric_cols].corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Health Metrics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_plotly():
    fig = px.scatter(health_data, x='bmi', y='systolic_bp', 
                     color='diagnosis', size='age',
                     hover_data=['patient_id', 'lab_result', 'outcome_score'],
                     title='Interactive Health Patient Dashboard')
    fig.update_layout(height=600)
    return json.dumps(fig, cls=PlotlyJSONEncoder)

# Education Visualization Generators
def generate_education_bar():
    subject_avg = education_data.groupby('subject')['exam_score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=subject_avg.index, y=subject_avg.values, palette='muted')
    plt.title('Average Exam Scores by Subject', fontsize=16, fontweight='bold')
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Average Exam Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_education_scatter():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=education_data, x='homework_hours', y='exam_score', 
                    hue='program', s=100, alpha=0.7, palette='Set2')
    plt.title('Homework Hours vs Exam Score by Program', fontsize=16, fontweight='bold')
    plt.xlabel('Homework Hours per Week', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.legend(title='Program')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_education_boxplot():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=education_data, x='program', y='exam_score', palette='pastel')
    plt.title('Exam Score Distribution by Program', fontsize=16, fontweight='bold')
    plt.xlabel('Academic Program', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_education_line():
    grade_avg = education_data.groupby('grade_level')['exam_score'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(grade_avg['grade_level'], grade_avg['exam_score'], marker='o', linewidth=2, markersize=10, color='steelblue')
    plt.title('Average Exam Score by Grade Level', fontsize=16, fontweight='bold')
    plt.xlabel('Grade Level', fontsize=12)
    plt.ylabel('Average Exam Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(grade_avg['grade_level'])
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_education_bokeh():
    p = figure(title='Interactive Student Performance Dashboard', 
               x_axis_label='Homework Hours', 
               y_axis_label='Exam Score',
               width=800, height=600)
    
    programs = education_data['program'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, program in enumerate(programs):
        program_data = education_data[education_data['program'] == program]
        p.scatter(program_data['homework_hours'], program_data['exam_score'],
                 size=10, color=colors[i % len(colors)], alpha=0.7, legend_label=program)
    
    p.legend.location = 'top_left'
    script, div = components(p, CDN)
    return {'script': script, 'div': div}

# Finance Visualization Generators
def generate_finance_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(finance_data['interest_rate'], kde=True, bins=10, color='gold', edgecolor='black')
    plt.title('Interest Rate Distribution - Mortgage Loans', fontsize=16, fontweight='bold')
    plt.xlabel('Interest Rate (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(finance_data['interest_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {finance_data["interest_rate"].mean():.1f}%')
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_finance_scatter():
    finance_data['defaulted_label'] = finance_data['defaulted'].map({0: 'No Default', 1: 'Defaulted'})
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=finance_data, x='interest_rate', y='dti_ratio', 
                    hue='defaulted_label', s=100, alpha=0.7, palette='RdYlGn_r')
    plt.title('Risk Analysis: Interest Rate vs Debt-to-Income Ratio', fontsize=16, fontweight='bold')
    plt.xlabel('Interest Rate (%)', fontsize=12)
    plt.ylabel('Debt-to-Income Ratio', fontsize=12)
    plt.legend(title='Loan Status')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_finance_boxplot():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=finance_data, x='bank', y='interest_rate', palette='Set3')
    plt.title('Interest Rate Comparison by Bank', fontsize=16, fontweight='bold')
    plt.xlabel('Bank', fontsize=12)
    plt.ylabel('Interest Rate (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_finance_heatmap():
    numeric_cols = ['interest_rate', 'term_years', 'principal', 'monthly_payment', 
                    'income', 'dti_ratio', 'credit_score']
    corr_matrix = finance_data[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Financial Metrics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_finance_plotly():
    fig = px.scatter(finance_data, x='interest_rate', y='dti_ratio', 
                     color='defaulted', size='principal',
                     hover_data=['bank', 'credit_score', 'region'],
                     title='Interactive Finance Loan Dashboard',
                     labels={'defaulted': 'Defaulted'})
    fig.update_layout(height=600)
    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/student-study')
def student_study():
    return render_template('student_study.html',
                         datasets=[
                             {'name': 'Scatter Plot with Trend Line', 'type': 'scatter', 'library': 'matplotlib'},
                             {'name': 'Line Plot', 'type': 'line', 'library': 'matplotlib'},
                             {'name': 'Regression Plot', 'type': 'regression', 'library': 'seaborn'},
                             {'name': 'Bar Plot Comparison', 'type': 'bar', 'library': 'seaborn'}
                         ])

@app.route('/student-study/<viz_type>')
def student_study_viz(viz_type):
    if viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='Student Study Success',
                             viz_name='Scatter Plot with Trend Line',
                             viz_type='Scatter Plot',
                             library='Matplotlib',
                             description='This scatter plot visualizes the relationship between study hours and success scores. Each point represents a student, with study hours on the x-axis and success scores on the y-axis. A trend line (linear regression) is added to show the overall relationship pattern. This is the most ideal visualization for answering "Does success increase as study hours increase?"',
                             lesson='Scatter plots with trend lines are perfect for identifying relationships between two continuous variables. Matplotlib provides excellent control for creating scatter plots with custom trend lines using numpy polyfit. This visualization clearly shows a positive correlation: as study hours increase, success scores tend to increase as well. The trend line helps visualize the strength and direction of this relationship.',
                             plot_data=generate_student_scatter())
    
    elif viz_type == 'line':
        return render_template('visualization.html',
                             domain='Student Study Success',
                             viz_name='Line Plot',
                             viz_type='Line Plot',
                             library='Matplotlib',
                             description='This line plot shows students ordered by study hours (from least to most). The line connects each student\'s data point, making it easy to see the progression of success scores as study hours increase. Students are arranged along the x-axis by their study hours.',
                             lesson='Line plots are useful for showing trends and progressions. In this case, ordering students by study hours allows us to see how success scores change as study time increases. Matplotlib\'s line plot is ideal for this type of sequential data visualization, making patterns and trends immediately visible.',
                             plot_data=generate_student_line())
    
    elif viz_type == 'regression':
        return render_template('visualization.html',
                             domain='Student Study Success',
                             viz_name='Regression Plot',
                             viz_type='Regression Plot',
                             library='Seaborn',
                             description='This regression plot combines a scatter plot with an automatically calculated trend line and confidence interval. Seaborn\'s regplot function automatically fits a regression model and displays the relationship between study hours and success scores. The shaded area represents the confidence interval around the regression line.',
                             lesson='Seaborn\'s regplot is specifically designed for visualizing relationships between variables. It automatically calculates and displays the regression line, making it the best choice for understanding correlations. The confidence interval (shaded area) shows the uncertainty in the regression estimate. This is the most comprehensive visualization for understanding the study hours-success relationship.',
                             plot_data=generate_student_regression())
    
    elif viz_type == 'bar':
        return render_template('visualization.html',
                             domain='Student Study Success',
                             viz_name='Bar Plot Comparison',
                             viz_type='Bar Plot',
                             library='Seaborn',
                             description='This bar plot shows each student\'s study hours and success score side by side. Each student has two bars: one for study hours and one for success score. This categorical view makes it easy to compare individual students and see both metrics simultaneously.',
                             lesson='Bar plots are excellent for comparing categorical data. In this case, we use a grouped bar plot to show both study hours and success scores for each student. Seaborn makes it easy to create grouped bar plots with proper labeling and color coding. While not ideal for showing correlations, bar plots are great for individual comparisons and understanding each student\'s profile.',
                             plot_data=generate_student_bar())
    
    else:
        return "Visualization type not found", 404

# Student Study Success Visualization Generators
def generate_student_scatter():
    
    # Sort by study hours for better visualization
    sorted_data = student_data.sort_values('study_hours')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_data['study_hours'], sorted_data['success_score'], 
                s=100, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(sorted_data['study_hours'], sorted_data['success_score'], 1)
    p = np.poly1d(z)
    plt.plot(sorted_data['study_hours'], p(sorted_data['study_hours']), 
             "r--", linewidth=2, label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Add student names as annotations
    for idx, row in sorted_data.iterrows():
        plt.annotate(row['student_name'], 
                    (row['study_hours'], row['success_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Study Hours vs Success Score (Scatter Plot with Trend Line)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Success Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_student_line():
    # Sort by study hours
    sorted_data = student_data.sort_values('study_hours').reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_data)), sorted_data['success_score'], 
             marker='o', linewidth=2, markersize=10, color='steelblue', label='Success Score')
    plt.plot(range(len(sorted_data)), sorted_data['study_hours'] * 10, 
             marker='s', linewidth=2, markersize=8, color='coral', 
             linestyle='--', label='Study Hours (×10 for scale)')
    
    plt.xticks(range(len(sorted_data)), sorted_data['student_name'], rotation=45)
    plt.title('Student Performance: Ordered by Study Hours', fontsize=16, fontweight='bold')
    plt.xlabel('Students (ordered by study hours)', fontsize=12)
    plt.ylabel('Score / Study Hours (×10)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_student_regression():
    plt.figure(figsize=(10, 6))
    sns.regplot(data=student_data, x='study_hours', y='success_score', 
                scatter_kws={'s': 100, 'alpha': 0.7, 'color': 'steelblue', 
                            'edgecolors': 'black', 'linewidths': 1.5},
                line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'})
    
    # Add student names
    for idx, row in student_data.iterrows():
        plt.annotate(row['student_name'], 
                    (row['study_hours'], row['success_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Study Hours vs Success Score (Regression Plot)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Success Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_student_bar():
    # Prepare data for grouped bar plot
    sorted_data = student_data.sort_values('study_hours')
    x = np.arange(len(sorted_data))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, sorted_data['study_hours'], width, 
                   label='Study Hours', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, sorted_data['success_score'] / 10, width, 
                   label='Success Score (÷10)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Students', fontsize=12)
    ax.set_ylabel('Study Hours / Success Score (÷10)', fontsize=12)
    ax.set_title('Student Study Hours and Success Score Comparison', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_data['student_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*10:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/home-loans')
def home_loans():
    return render_template('home_loans.html',
                         datasets=[
                             {'name': 'Scatter Plot Analysis', 'type': 'scatter', 'library': 'seaborn'},
                             {'name': 'Regression Plot', 'type': 'regression', 'library': 'seaborn'},
                             {'name': 'Distribution Analysis', 'type': 'histogram', 'library': 'seaborn'},
                             {'name': 'Box Plot Comparison', 'type': 'boxplot', 'library': 'seaborn'},
                             {'name': 'Correlation Heatmap', 'type': 'heatmap', 'library': 'seaborn'},
                             {'name': 'Bar Chart Analysis', 'type': 'bar', 'library': 'seaborn'}
                         ])

@app.route('/home-loans/<viz_type>')
def home_loans_viz(viz_type):
    if viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='Home Loan Applicants',
                             viz_name='Scatter Plot Analysis',
                             viz_type='Scatter Plot',
                             library='Seaborn',
                             description='This scatter plot visualizes the relationship between credit score and interest rate. In mortgage analysis, understanding this relationship is crucial: as credit scores increase, interest rates typically decrease. This visualization helps answer "Does interest rate decrease as credit score increases?"',
                             lesson='Scatter plots are essential in mortgage underwriting for identifying relationships between key financial metrics. Seaborn\'s scatterplot provides beautiful default styling and easy customization. This visualization clearly shows the inverse relationship between credit scores and interest rates - higher credit scores lead to lower interest rates, which is a fundamental principle in mortgage lending.',
                             plot_data=generate_loan_scatter())
    
    elif viz_type == 'regression':
        return render_template('visualization.html',
                             domain='Home Loan Applicants',
                             viz_name='Regression Plot',
                             viz_type='Regression Plot',
                             library='Seaborn',
                             description='This regression plot shows the relationship between income and debt-to-income (DTI) ratio with an automatically calculated trend line and confidence interval. It helps understand if higher income correlates with lower DTI ratios, which is important for loan risk assessment.',
                             lesson='Regression plots are super useful in mortgage analysis. Seaborn\'s regplot automatically calculates and displays the regression line with confidence intervals, making it perfect for understanding correlations. This visualization shows that as income increases, DTI ratios tend to decrease, indicating better financial health and lower loan risk.',
                             plot_data=generate_loan_regression())
    
    elif viz_type == 'histogram':
        return render_template('visualization.html',
                             domain='Home Loan Applicants',
                             viz_name='Distribution Analysis',
                             viz_type='Histogram with KDE',
                             library='Seaborn',
                             description='This histogram with Kernel Density Estimation (KDE) shows the distribution of interest rates across loan applicants. Understanding interest rate distributions is one of the most commonly used visualizations in mortgage underwriting, helping identify typical rates and outliers.',
                             lesson='Histograms with KDE are fundamental in mortgage underwriting for understanding the distribution of key metrics like interest rates, DTI ratios, and income. Seaborn\'s histplot with KDE provides a smooth estimate of the probability distribution, making it easy to identify normal ranges, outliers, and distribution patterns. This helps underwriters assess risk and set appropriate lending criteria.',
                             plot_data=generate_loan_histogram())
    
    elif viz_type == 'boxplot':
        return render_template('visualization.html',
                             domain='Home Loan Applicants',
                             viz_name='Box Plot Comparison',
                             viz_type='Box Plot',
                             library='Seaborn',
                             description='This box plot compares interest rates across different credit score ranges. Box plots show quartiles, medians, and outliers, making it easy to see how interest rates vary by credit score category. This is ideal for comparing rates across different risk groups.',
                             lesson='Box plots are excellent in mortgage analysis for comparing metrics across different groups. They reveal distribution differences, medians, and outliers, helping lenders understand how interest rates vary by credit score ranges. Seaborn\'s boxplot makes it easy to create these comparisons with beautiful default styling. This visualization helps identify risk-based pricing patterns.',
                             plot_data=generate_loan_boxplot())
    
    elif viz_type == 'heatmap':
        return render_template('visualization.html',
                             domain='Home Loan Applicants',
                             viz_name='Correlation Heatmap',
                             viz_type='Correlation Heatmap',
                             library='Seaborn',
                             description='This correlation heatmap shows relationships between all numeric variables: income, DTI ratio, credit score, and interest rate. This is the most critical visualization in mortgage underwriting, as it reveals how all key metrics relate to each other.',
                             lesson='Correlation heatmaps are the most critical visualization in mortgage underwriting. They reveal relationships between income, DTI ratio, credit score, and interest rate all at once. Seaborn\'s heatmap with correlation matrices makes it easy to identify strong positive or negative correlations. Understanding these relationships helps underwriters assess overall loan risk and make informed lending decisions.',
                             plot_data=generate_loan_heatmap())
    
    elif viz_type == 'bar':
        return render_template('visualization.html',
                             domain='Home Loan Applicants',
                             viz_name='Bar Chart Analysis',
                             viz_type='Bar Plot',
                             library='Seaborn',
                             description='This bar chart shows average interest rates by credit score ranges. Bar charts are useful for comparing averages across categories, making it easy to see how interest rates differ by credit score groups.',
                             lesson='Bar charts in mortgage analysis help compare averages across different groups. Seaborn\'s barplot provides statistical aggregation and error bars, making it perfect for comparing average interest rates across credit score ranges. This visualization helps lenders understand pricing strategies and helps borrowers understand how credit scores affect their rates.',
                             plot_data=generate_loan_bar())
    
    else:
        return "Visualization type not found", 404

# Home Loan Applicants Visualization Generators
def generate_loan_scatter():
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Credit Score vs Interest Rate
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=loan_data, x='credit_score', y='interest_rate', 
                    s=100, alpha=0.7, color='steelblue', edgecolors='black', linewidths=1.5)
    # Add trend line
    z = np.polyfit(loan_data['credit_score'], loan_data['interest_rate'], 1)
    p = np.poly1d(z)
    plt.plot(loan_data['credit_score'], p(loan_data['credit_score']), 
             "r--", linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
    plt.xlabel('Credit Score', fontsize=11)
    plt.ylabel('Interest Rate (%)', fontsize=11)
    plt.title('Credit Score vs Interest Rate', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Income vs DTI Ratio
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=loan_data, x='income', y='dti_ratio', 
                    s=100, alpha=0.7, color='coral', edgecolors='black', linewidths=1.5)
    # Add trend line
    z2 = np.polyfit(loan_data['income'], loan_data['dti_ratio'], 1)
    p2 = np.poly1d(z2)
    plt.plot(loan_data['income'], p2(loan_data['income']), 
             "r--", linewidth=2, label=f'Trend: y={z2[0]:.2e}x+{z2[1]:.2f}')
    plt.xlabel('Income ($)', fontsize=11)
    plt.ylabel('DTI Ratio', fontsize=11)
    plt.title('Income vs DTI Ratio', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Mortgage Scatter Plot Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_loan_regression():
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Credit Score vs Interest Rate
    plt.subplot(1, 2, 1)
    sns.regplot(data=loan_data, x='credit_score', y='interest_rate',
                scatter_kws={'s': 100, 'alpha': 0.7, 'color': 'steelblue', 
                            'edgecolors': 'black', 'linewidths': 1.5},
                line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'})
    plt.xlabel('Credit Score', fontsize=11)
    plt.ylabel('Interest Rate (%)', fontsize=11)
    plt.title('Credit Score vs Interest Rate (Regression)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Income vs DTI Ratio
    plt.subplot(1, 2, 2)
    sns.regplot(data=loan_data, x='income', y='dti_ratio',
                scatter_kws={'s': 100, 'alpha': 0.7, 'color': 'coral', 
                            'edgecolors': 'black', 'linewidths': 1.5},
                line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'})
    plt.xlabel('Income ($)', fontsize=11)
    plt.ylabel('DTI Ratio', fontsize=11)
    plt.title('Income vs DTI Ratio (Regression)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Mortgage Regression Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_loan_histogram():
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Interest Rate Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(loan_data['interest_rate'], kde=True, bins=8, color='steelblue', edgecolor='black')
    plt.xlabel('Interest Rate (%)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Interest Rate Distribution', fontsize=13, fontweight='bold')
    plt.axvline(loan_data['interest_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {loan_data["interest_rate"].mean():.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: DTI Ratio Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(loan_data['dti_ratio'], kde=True, bins=8, color='coral', edgecolor='black')
    plt.xlabel('DTI Ratio', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('DTI Ratio Distribution', fontsize=13, fontweight='bold')
    plt.axvline(loan_data['dti_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {loan_data["dti_ratio"].mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Income Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(loan_data['income'], kde=True, bins=8, color='lightgreen', edgecolor='black')
    plt.xlabel('Income ($)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Income Distribution', fontsize=13, fontweight='bold')
    plt.axvline(loan_data['income'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${loan_data["income"].mean():,.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Credit Score Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(loan_data['credit_score'], kde=True, bins=8, color='gold', edgecolor='black')
    plt.xlabel('Credit Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Credit Score Distribution', fontsize=13, fontweight='bold')
    plt.axvline(loan_data['credit_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {loan_data["credit_score"].mean():.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Mortgage Metrics Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_loan_boxplot():
    # Create credit score categories
    loan_data_copy = loan_data.copy()
    loan_data_copy['credit_category'] = pd.cut(loan_data_copy['credit_score'], 
                                                bins=[0, 650, 700, 750, 850],
                                                labels=['Low (640-650)', 'Medium (650-700)', 
                                                       'Good (700-750)', 'Excellent (750+)'])
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Interest Rate by Credit Score Category
    plt.subplot(1, 2, 1)
    sns.boxplot(data=loan_data_copy, x='credit_category', y='interest_rate', palette='Set2')
    plt.xlabel('Credit Score Category', fontsize=11)
    plt.ylabel('Interest Rate (%)', fontsize=11)
    plt.title('Interest Rate by Credit Score Category', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: DTI Ratio by Income Groups
    loan_data_copy['income_group'] = pd.cut(loan_data_copy['income'], 
                                           bins=[0, 60000, 80000, 100000, 150000],
                                           labels=['Low (<60k)', 'Medium (60-80k)', 
                                                  'High (80-100k)', 'Very High (100k+)'])
    sns.boxplot(data=loan_data_copy, x='income_group', y='dti_ratio', palette='Set3')
    plt.xlabel('Income Group', fontsize=11)
    plt.ylabel('DTI Ratio', fontsize=11)
    plt.title('DTI Ratio by Income Group', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Mortgage Box Plot Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_loan_heatmap():
    numeric_cols = ['income', 'dti_ratio', 'credit_score', 'interest_rate']
    corr_matrix = loan_data[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1)
    plt.title('Mortgage Metrics Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_loan_bar():
    # Create credit score ranges
    loan_data_copy = loan_data.copy()
    loan_data_copy['credit_range'] = pd.cut(loan_data_copy['credit_score'], 
                                           bins=[0, 650, 700, 750, 850],
                                           labels=['640-650', '650-700', '700-750', '750+'])
    
    avg_rates = loan_data_copy.groupby('credit_range')['interest_rate'].mean().sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(avg_rates)), avg_rates.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(avg_rates))), alpha=0.8)
    plt.xlabel('Credit Score Range', fontsize=12)
    plt.ylabel('Average Interest Rate (%)', fontsize=12)
    plt.title('Average Interest Rate by Credit Score Range', fontsize=16, fontweight='bold')
    plt.xticks(range(len(avg_rates)), avg_rates.index)
    
    # Add value labels on bars
    for i, v in enumerate(avg_rates.values):
        plt.text(i, v + 0.05, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/ecommerce-csv')
def ecommerce_csv():
    return render_template('ecommerce_csv.html',
                         datasets=[
                             {'name': 'Sales Trend Analysis', 'type': 'line', 'library': 'matplotlib'},
                             {'name': 'Category Sales Comparison', 'type': 'bar', 'library': 'seaborn'},
                             {'name': 'Cart Value Distribution', 'type': 'histogram', 'library': 'seaborn'},
                             {'name': 'Traffic Source Performance', 'type': 'countplot', 'library': 'seaborn'},
                             {'name': 'Country-Device Heatmap', 'type': 'heatmap', 'library': 'seaborn'},
                             {'name': 'Price-Quantity Scatter', 'type': 'scatter', 'library': 'seaborn'},
                             {'name': 'Multi-Variable Pairplot', 'type': 'pairplot', 'library': 'seaborn'},
                             {'name': 'Device Sales Distribution', 'type': 'pie', 'library': 'matplotlib'}
                         ])

@app.route('/ecommerce-csv/<viz_type>')
def ecommerce_csv_viz(viz_type):
    if viz_type == 'line':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Sales Trend Analysis',
                             viz_type='Line Plot',
                             library='Matplotlib',
                             description='This line plot shows daily total sales (revenue) over time. It visualizes how revenue changes over time, helping identify seasonal trends, peak sales days, and overall business growth patterns. The x-axis shows order dates and the y-axis displays total daily revenue.',
                             lesson='Line plots are essential for time series analysis in e-commerce. Matplotlib provides precise control for creating line plots that show revenue trends over time. This visualization helps businesses understand sales performance, identify peak periods, and forecast future revenue. Technology: Matplotlib (plot) or Seaborn (lineplot).',
                             plot_data=generate_ecommerce_csv_line())
    
    elif viz_type == 'bar':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Category Sales Comparison',
                             viz_type='Bar Plot',
                             library='Seaborn',
                             description='This bar chart compares total revenue across different product categories. It helps identify which categories generate the most sales and which product classes have the highest profit potential. Each bar represents a category, showing total revenue.',
                             lesson='Bar plots are fundamental in e-commerce analytics for comparing performance across categories. Seaborn\'s barplot provides statistical aggregation and beautiful default styling. This visualization helps businesses identify top-performing categories and allocate resources accordingly. Technology: Seaborn (barplot) or Matplotlib (bar).',
                             plot_data=generate_ecommerce_csv_bar())
    
    elif viz_type == 'histogram':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Cart Value Distribution',
                             viz_type='Histogram with KDE',
                             library='Seaborn',
                             description='This histogram with KDE shows the distribution of cart values (price × quantity). It helps understand customer spending behavior, identify average spending amounts, and detect outlier purchases. The KDE curve provides a smooth estimate of the probability distribution.',
                             lesson='Histograms with KDE are powerful tools for understanding data distributions in e-commerce. They reveal customer behavior patterns, such as typical spending amounts and whether the distribution is normal, skewed, or has multiple peaks. Seaborn\'s histplot with KDE is the most suitable tool for this analysis. Technology: Seaborn (histplot, kdeplot) - most suitable, or Matplotlib (hist).',
                             plot_data=generate_ecommerce_csv_histogram())
    
    elif viz_type == 'countplot':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Traffic Source Performance',
                             viz_type='Count Plot',
                             library='Seaborn',
                             description='This count plot shows the number of orders from each traffic source. It helps identify which marketing channels bring in the most sales, enabling businesses to optimize their marketing budget allocation.',
                             lesson='Count plots are excellent for showing frequency distributions of categorical data. Seaborn\'s countplot automatically counts occurrences and creates bar charts, making it perfect for analyzing traffic source performance. This visualization helps businesses understand which marketing channels are most effective. Technology: Seaborn (countplot) or Matplotlib (bar).',
                             plot_data=generate_ecommerce_csv_countplot())
    
    elif viz_type == 'heatmap':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Country-Device Heatmap',
                             viz_type='Heatmap',
                             library='Seaborn',
                             description='This heatmap displays the relationship between countries and device types used for purchases. It shows sales density across country-device combinations, helping identify patterns like whether mobile is dominant in USA or desktop is higher in Europe.',
                             lesson='Heatmaps are excellent for visualizing relationships between two categorical variables in e-commerce. Seaborn\'s heatmap combined with Pandas pivot_table makes it easy to create cross-tabulation visualizations. This helps identify market segments and understand customer preferences by region and device type. Technology: Seaborn (heatmap) and Pandas (pivot_table).',
                             plot_data=generate_ecommerce_csv_heatmap())
    
    elif viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Price-Quantity Scatter',
                             viz_type='Scatter Plot',
                             library='Seaborn',
                             description='This scatter plot explores the relationship between price and quantity, colored by category. It helps answer questions like "Do cheaper products sell more?" and "How is price elasticity for Fashion products?"',
                             lesson='Scatter plots are excellent for relationship analysis in e-commerce. Seaborn\'s scatterplot provides beautiful default styling and easy color coding by category. This visualization helps understand pricing strategies, demand elasticity, and category-specific patterns. Technology: Seaborn (scatterplot) or Matplotlib (scatter).',
                             plot_data=generate_ecommerce_csv_scatter())
    
    elif viz_type == 'pairplot':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Multi-Variable Pairplot',
                             viz_type='Pair Plot',
                             library='Seaborn',
                             description='This pair plot shows relationships between multiple variables (price, quantity, revenue) simultaneously, with categories color-coded. It provides a comprehensive view of all variable relationships in one visualization, making it ideal for multi-variable analysis.',
                             lesson='Pair plots are unique to Seaborn and provide an excellent way to analyze multiple variables at once. They show scatter plots for all variable pairs, histograms on the diagonal, and can be color-coded by category. This is the most comprehensive visualization for understanding complex relationships in e-commerce data. Technology: Seaborn (pairplot) - not available in Matplotlib!',
                             plot_data=generate_ecommerce_csv_pairplot())
    
    elif viz_type == 'pie':
        return render_template('visualization.html',
                             domain='E-Commerce Sales',
                             viz_name='Device Sales Distribution',
                             viz_type='Pie Chart',
                             library='Matplotlib',
                             description='This pie chart shows the distribution of sales by device type (Mobile, Desktop, Tablet). It helps answer "Does mobile or desktop bring more sales?" by showing the proportion of sales from each device type.',
                             lesson='Pie charts are useful for showing proportions and percentages. Matplotlib provides excellent control for creating pie charts with custom styling. Bar charts are often preferred for exact comparisons, but pie charts provide an intuitive view of proportions. Technology: Matplotlib (pie, bar) or Seaborn (barplot).',
                             plot_data=generate_ecommerce_csv_pie())
    
    else:
        return "Visualization type not found", 404

# E-Commerce CSV Visualization Generators
def generate_ecommerce_csv_line():
    daily_revenue = ecommerce_csv_data.groupby('order_date')['revenue'].sum().reset_index()
    daily_revenue = daily_revenue.sort_values('order_date')
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_revenue['order_date'], daily_revenue['revenue'], 
             marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.title('Daily Sales Trend Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Order Date', fontsize=12)
    plt.ylabel('Total Daily Revenue ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_bar():
    category_revenue = ecommerce_csv_data.groupby('category')['revenue'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_revenue.index, y=category_revenue.values, palette='viridis', 
                hue=category_revenue.index, legend=False)
    plt.title('Category Sales Comparison - Total Revenue by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Revenue ($)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(category_revenue.values):
        plt.text(i, v + 500, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(ecommerce_csv_data['revenue'], kde=True, bins=15, 
                color='skyblue', edgecolor='black')
    plt.title('Cart Value Distribution (Price × Quantity)', fontsize=16, fontweight='bold')
    plt.xlabel('Cart Value / Revenue ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(ecommerce_csv_data['revenue'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${ecommerce_csv_data["revenue"].mean():,.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_countplot():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=ecommerce_csv_data, x='traffic_source', 
                  palette='Set2', hue='traffic_source', legend=False)
    plt.title('Traffic Source Performance - Order Count by Channel', fontsize=16, fontweight='bold')
    plt.xlabel('Traffic Source', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_heatmap():
    pivot_data = ecommerce_csv_data.groupby(['country', 'device']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Orders'}, linewidths=1)
    plt.title('Country-Device Sales Density Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Device Type', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_scatter():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=ecommerce_csv_data, x='price', y='quantity', 
                    hue='category', s=100, alpha=0.7, palette='Set1')
    plt.title('Price vs Quantity by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Quantity', fontsize=12)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_pairplot():
    # Select numeric columns for pairplot
    pairplot_data = ecommerce_csv_data[['price', 'quantity', 'revenue', 'category']]
    
    # Create pairplot
    g = sns.pairplot(pairplot_data, hue='category', palette='Set1', 
                    diag_kind='kde', height=2.5)
    g.fig.suptitle('Multi-Variable Relationship Analysis (Pair Plot)', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    img = io.BytesIO()
    g.figure.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plt.close('all')
    return base64.b64encode(img.getvalue()).decode()

def generate_ecommerce_csv_pie():
    device_counts = ecommerce_csv_data['device'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax1.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=(0.05, 0.05, 0.05))
    ax1.set_title('Device Sales Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Bar chart for comparison
    bars = ax2.bar(device_counts.index, device_counts.values, 
                   color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Device Sales Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Device Type', fontsize=12)
    ax2.set_ylabel('Number of Orders', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Device-Based Sales Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/health-clinic')
def health_clinic():
    return render_template('health_clinic.html',
                         datasets=[
                             {'name': 'Age/BMI/Blood Pressure Distribution', 'type': 'histogram', 'library': 'seaborn'},
                             {'name': 'Diagnosis-Based Comparison', 'type': 'boxplot', 'library': 'seaborn'},
                             {'name': 'BMI vs Blood Pressure Relationship', 'type': 'scatter', 'library': 'seaborn'},
                             {'name': 'Health Metrics Correlation', 'type': 'heatmap', 'library': 'seaborn'},
                             {'name': 'Hospital Treatment Success', 'type': 'bar', 'library': 'seaborn'}
                         ])

@app.route('/health-clinic/<viz_type>')
def health_clinic_viz(viz_type):
    if viz_type == 'histogram':
        return render_template('visualization.html',
                             domain='Health Clinic Data',
                             viz_name='Age/BMI/Blood Pressure Distribution',
                             viz_type='Histogram with KDE',
                             library='Seaborn',
                             description='This histogram with KDE shows the distribution of age, BMI, and systolic blood pressure across all patients. Understanding these distributions is essential for identifying normal ranges, outliers, and population health characteristics.',
                             lesson='Histograms with KDE are the best tool for statistical distribution analysis in healthcare. Seaborn\'s histplot and kdeplot provide smooth probability distribution estimates, making it easy to identify normal ranges, outliers, and distribution patterns. This visualization helps healthcare providers understand patient population characteristics and plan preventive care strategies. Technology: Seaborn (histplot, kdeplot) - best for statistical distribution.',
                             plot_data=generate_health_clinic_histogram())
    
    elif viz_type == 'boxplot':
        return render_template('visualization.html',
                             domain='Health Clinic Data',
                             viz_name='Diagnosis-Based Comparison',
                             viz_type='Box Plot / Violin Plot',
                             library='Seaborn',
                             description='This visualization compares BMI and A1C lab results across different diagnoses using box plots and violin plots. Box plots show quartiles, medians, and outliers, while violin plots show the full distribution shape. This helps identify which conditions are associated with higher or lower values.',
                             lesson='Box plots and violin plots are excellent for comparing patient metrics across different diagnoses. They reveal distribution differences, medians, and outliers, making it visually clear how conditions affect health metrics. Seaborn\'s boxplot and violinplot provide beautiful default styling and make outliers immediately apparent. Technology: Seaborn (boxplot, violinplot) - visually clear, outliers are obvious.',
                             plot_data=generate_health_clinic_boxplot())
    
    elif viz_type == 'scatter':
        return render_template('visualization.html',
                             domain='Health Clinic Data',
                             viz_name='BMI vs Blood Pressure Relationship',
                             viz_type='Scatter Plot with Regression',
                             library='Seaborn',
                             description='This scatter plot explores the relationship between BMI and systolic blood pressure, colored by diagnosis. A regression line is automatically calculated and displayed with confidence intervals. This helps identify correlations and risk factors.',
                             lesson='Scatter plots with regression lines are fundamental in healthcare research for identifying relationships between health metrics. Seaborn\'s scatterplot and regplot make it easy to visualize correlations and understand how BMI affects blood pressure. Optionally, Plotly can be used for interactive visualizations with hover details showing patient_id. Technology: Seaborn (scatterplot, regplot) or optionally Plotly for interactive (hover: patient_id).',
                             plot_data=generate_health_clinic_scatter())
    
    elif viz_type == 'heatmap':
        return render_template('visualization.html',
                             domain='Health Clinic Data',
                             viz_name='Health Metrics Correlation',
                             viz_type='Correlation Heatmap',
                             library='Seaborn',
                             description='This correlation heatmap shows relationships between all numeric health metrics: age, BMI, systolic BP, diastolic BP, A1C lab results, and outcome scores. Understanding these correlations helps identify risk factors and predict outcomes.',
                             lesson='Correlation heatmaps are essential in healthcare analytics for understanding how different health metrics relate to each other. Seaborn\'s heatmap with correlation matrices makes it easy to identify strong positive or negative correlations. While static, this visualization is perfect for analysis and helps guide treatment decisions. Technology: Seaborn (heatmap) - static but perfect for analysis.',
                             plot_data=generate_health_clinic_heatmap())
    
    elif viz_type == 'bar':
        return render_template('visualization.html',
                             domain='Health Clinic Data',
                             viz_name='Hospital Treatment Success',
                             viz_type='Bar Plot',
                             library='Seaborn',
                             description='This bar chart compares average outcome scores across different hospitals. It helps identify which hospitals have better treatment success rates and can inform patient referrals and quality improvement initiatives.',
                             lesson='Bar charts in healthcare help compare performance across different hospitals, treatments, or time periods. Seaborn\'s barplot provides statistical aggregation and error bars, making it perfect for comparing average outcome scores. Optionally, Dash or Plotly can be used to create interactive dashboards for real-time monitoring. Technology: Seaborn (barplot) or optionally Dash/Plotly for interactive dashboard.',
                             plot_data=generate_health_clinic_bar())
    
    else:
        return "Visualization type not found", 404

# Health Clinic Visualization Generators
def generate_health_clinic_histogram():
    plt.figure(figsize=(14, 4))
    
    # Subplot 1: Age Distribution
    plt.subplot(1, 3, 1)
    sns.histplot(health_clinic_data['age'], kde=True, bins=8, color='steelblue', edgecolor='black')
    plt.xlabel('Age', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Age Distribution', fontsize=13, fontweight='bold')
    plt.axvline(health_clinic_data['age'].mean(), color='red', linestyle='--', 
                label=f'Mean: {health_clinic_data["age"].mean():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: BMI Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(health_clinic_data['bmi'], kde=True, bins=8, color='coral', edgecolor='black')
    plt.xlabel('BMI', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('BMI Distribution', fontsize=13, fontweight='bold')
    plt.axvline(health_clinic_data['bmi'].mean(), color='red', linestyle='--', 
                label=f'Mean: {health_clinic_data["bmi"].mean():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Systolic BP Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(health_clinic_data['systolic_bp'], kde=True, bins=8, color='lightgreen', edgecolor='black')
    plt.xlabel('Systolic Blood Pressure (mmHg)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Systolic BP Distribution', fontsize=13, fontweight='bold')
    plt.axvline(health_clinic_data['systolic_bp'].mean(), color='red', linestyle='--', 
                label=f'Mean: {health_clinic_data["systolic_bp"].mean():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Age / BMI / Blood Pressure Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_clinic_boxplot():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: BMI by Diagnosis (Box Plot)
    sns.boxplot(data=health_clinic_data, x='diagnosis', y='bmi', palette='Set2', ax=axes[0])
    axes[0].set_xlabel('Diagnosis', fontsize=12)
    axes[0].set_ylabel('BMI', fontsize=12)
    axes[0].set_title('BMI by Diagnosis (Box Plot)', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: A1C by Diagnosis (Violin Plot)
    sns.violinplot(data=health_clinic_data, x='diagnosis', y='lab_a1c', palette='Set3', ax=axes[1])
    axes[1].set_xlabel('Diagnosis', fontsize=12)
    axes[1].set_ylabel('A1C Lab Result', fontsize=12)
    axes[1].set_title('A1C by Diagnosis (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Diagnosis-Based Health Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_clinic_scatter():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=health_clinic_data, x='bmi', y='systolic_bp', 
                    hue='diagnosis', s=100, alpha=0.7, palette='Set1')
    
    # Add regression line for each diagnosis group
    for diagnosis in health_clinic_data['diagnosis'].unique():
        group_data = health_clinic_data[health_clinic_data['diagnosis'] == diagnosis]
        if len(group_data) > 1:
            z = np.polyfit(group_data['bmi'], group_data['systolic_bp'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(group_data['bmi'].min(), group_data['bmi'].max(), 100)
            plt.plot(x_line, p(x_line), '--', alpha=0.6, linewidth=1.5)
    
    plt.title('BMI vs Systolic Blood Pressure by Diagnosis', fontsize=16, fontweight='bold')
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Systolic Blood Pressure (mmHg)', fontsize=12)
    plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_clinic_heatmap():
    numeric_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'lab_a1c', 'outcome_score']
    corr_matrix = health_clinic_data[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1)
    plt.title('Health Metrics Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_health_clinic_bar():
    hospital_outcomes = health_clinic_data.groupby('hospital')['outcome_score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(hospital_outcomes)), hospital_outcomes.values,
                   color=plt.cm.viridis(np.linspace(0, 1, len(hospital_outcomes))), alpha=0.8)
    plt.xlabel('Hospital', fontsize=12)
    plt.ylabel('Average Outcome Score', fontsize=12)
    plt.title('Hospital Treatment Success - Average Outcome Score by Hospital', 
              fontsize=16, fontweight='bold')
    plt.xticks(range(len(hospital_outcomes)), hospital_outcomes.index, rotation=15)
    
    # Add value labels on bars
    for i, v in enumerate(hospital_outcomes.values):
        plt.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/interactive-3d')
def interactive_3d():
    return render_template('interactive_3d.html',
                         datasets=[
                             {'name': '3D Scatter Plot (Static)', 'type': 'matplotlib3d', 'library': 'matplotlib'},
                             {'name': '3D Scatter Plot (Interactive)', 'type': 'plotly3d', 'library': 'plotly'},
                             {'name': '2D Projection Comparison', 'type': 'projection', 'library': 'seaborn'}
                         ])

@app.route('/interactive-3d/<viz_type>')
def interactive_3d_viz(viz_type):
    if viz_type == 'matplotlib3d':
        return render_template('visualization.html',
                             domain='Interactive 3D Dataset',
                             viz_name='3D Scatter Plot (Static)',
                             viz_type='3D Scatter Plot',
                             library='Matplotlib',
                             description='This 3D scatter plot shows the relationship between three features (feature_x, feature_y, feature_z) with points colored by risk category. Matplotlib 3D provides basic 3D visualization capabilities for static plots.',
                             lesson='Matplotlib 3D is the classic Python method for creating 3D visualizations. It can create 3D scatter plots, line plots, and surface plots. Advantages: Simple 3D plotting, most common classic method in Python. Disadvantages: Not interactive (only slight rotation in notebooks). Technology: Matplotlib (mpl_toolkits.mplot3d) - best for static 3D plots.',
                             plot_data=generate_3d_matplotlib())
    
    elif viz_type == 'plotly3d':
        return render_template('visualization.html',
                             domain='Interactive 3D Dataset',
                             viz_name='3D Scatter Plot (Interactive)',
                             viz_type='3D Interactive Plot',
                             library='Plotly',
                             description='This interactive 3D scatter plot allows you to rotate, zoom, and hover over points to see details. Points are colored by risk category and you can interact with the visualization in real-time. This is the most powerful option for 3D interactive visualizations.',
                             lesson='Plotly is the best technology for fully interactive 3D visualizations. It is excellent for creating real dashboards in healthcare, finance, and e-commerce. Advantages: Hover tooltips, zoom, rotate, click highlight, can export as HTML. This is the best technology for 3D + interactive visualizations. Technology: Plotly (scatter_3d) - best for interactive 3D with hover, zoom, and rotation.',
                             plot_data=generate_3d_plotly())
    
    elif viz_type == 'projection':
        return render_template('visualization.html',
                             domain='Interactive 3D Dataset',
                             viz_name='2D Projection Comparison',
                             viz_type='2D Scatter Plot',
                             library='Seaborn',
                             description='This visualization shows 2D projections of the 3D data, displaying different feature pairs (X-Y, X-Z, Y-Z) to help understand the relationships from different angles. Points are colored by risk category.',
                             lesson='2D projections are useful for understanding 3D data from different perspectives. While not truly 3D, they help visualize relationships between feature pairs. Seaborn provides beautiful scatter plots with easy color coding. Technology: Seaborn (scatterplot) - useful for 2D projections of 3D data.',
                             plot_data=generate_3d_projection())
    
    else:
        return "Visualization type not found", 404

# Interactive 3D Dataset Visualization Generators
def generate_3d_matplotlib():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping for risk categories
    color_map = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    
    for risk in interactive_3d_data['risk_category'].unique():
        data = interactive_3d_data[interactive_3d_data['risk_category'] == risk]
        ax.scatter(data['feature_x'], data['feature_y'], data['feature_z'],
                  c=color_map[risk], label=risk, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Feature X', fontsize=12)
    ax.set_ylabel('Feature Y', fontsize=12)
    ax.set_zlabel('Feature Z', fontsize=12)
    ax.set_title('3D Scatter Plot - Feature Relationships (Static)', fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def generate_3d_plotly():
    fig = px.scatter_3d(interactive_3d_data,
                       x='feature_x',
                       y='feature_y',
                       z='feature_z',
                       color='risk_category',
                       hover_name='id',
                       hover_data={'feature_x': True, 'feature_y': True, 'feature_z': True},
                       color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'},
                       title='Interactive 3D Scatter Plot - Feature Relationships',
                       labels={'feature_x': 'Feature X', 'feature_y': 'Feature Y', 'feature_z': 'Feature Z'})
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Feature X',
            yaxis_title='Feature Y',
            zaxis_title='Feature Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=700,
        width=1000,
        title_font_size=16,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def generate_3d_projection():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # X-Y Projection
    sns.scatterplot(data=interactive_3d_data, x='feature_x', y='feature_y', 
                   hue='risk_category', s=100, alpha=0.7, palette={'Low Risk': 'green', 
                   'Medium Risk': 'orange', 'High Risk': 'red'}, ax=axes[0])
    axes[0].set_title('X-Y Projection', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature X', fontsize=11)
    axes[0].set_ylabel('Feature Y', fontsize=11)
    axes[0].legend(title='Risk Category')
    axes[0].grid(True, alpha=0.3)
    
    # X-Z Projection
    sns.scatterplot(data=interactive_3d_data, x='feature_x', y='feature_z', 
                   hue='risk_category', s=100, alpha=0.7, palette={'Low Risk': 'green', 
                   'Medium Risk': 'orange', 'High Risk': 'red'}, ax=axes[1])
    axes[1].set_title('X-Z Projection', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature X', fontsize=11)
    axes[1].set_ylabel('Feature Z', fontsize=11)
    axes[1].legend(title='Risk Category')
    axes[1].grid(True, alpha=0.3)
    
    # Y-Z Projection
    sns.scatterplot(data=interactive_3d_data, x='feature_y', y='feature_z', 
                   hue='risk_category', s=100, alpha=0.7, palette={'Low Risk': 'green', 
                   'Medium Risk': 'orange', 'High Risk': 'red'}, ax=axes[2])
    axes[2].set_title('Y-Z Projection', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Feature Y', fontsize=11)
    axes[2].set_ylabel('Feature Z', fontsize=11)
    axes[2].legend(title='Risk Category')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('2D Projections of 3D Data - Feature Pair Relationships', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/visualization-guide')
def visualization_guide():
    return render_template('visualization_guide.html')

@app.route('/tools-metrics')
def tools_metrics():
    return render_template('tools_metrics.html')

@app.route('/capability-metrics')
def capability_metrics():
    return render_template('capability_metrics.html')

@app.route('/lessons')
def lessons():
    return render_template('lessons.html')

@app.route('/lesson/<library>')
def lesson_page(library):
    if library == 'matplotlib':
        return render_template('lesson.html',
                             library_name='Matplotlib',
                             overview='Matplotlib is Python\'s foundational plotting library, providing a MATLAB-like interface for creating static, publication-quality visualizations. It offers extensive customization options and is the base for many other visualization libraries.',
                             features=[
                                 {'title': 'Static Visualizations', 'description': 'Creates high-quality static plots perfect for publications, reports, and presentations.'},
                                 {'title': 'High Customization', 'description': 'Extensive control over every aspect of the plot including colors, fonts, axes, and annotations.'},
                                 {'title': 'Multiple Backends', 'description': 'Supports various output formats including PNG, PDF, SVG, and interactive backends.'},
                                 {'title': 'Wide Compatibility', 'description': 'Works seamlessly with NumPy, Pandas, and other scientific Python libraries.'}
                             ],
                             best_for=[
                                 'Time series analysis and trends',
                                 'Publication-quality figures',
                                 'Custom, highly-styled visualizations',
                                 'Scientific and research plots'
                             ],
                             domain_uses=[
                                 'E-Commerce: Daily revenue trends over time',
                                 'Education: Grade level performance progression',
                                 'Finance: Interest rate trends (when time data is available)',
                                 'Health: Patient metric tracking over time'
                             ],
                             examples=[
                                 {'name': 'Daily Revenue Trend', 'domain': 'E-Commerce', 'description': 'Line plot showing daily sales revenue over time', 'link': '/ecommerce/line'},
                                 {'name': 'Grade Level Performance', 'domain': 'Education', 'description': 'Line plot tracking average exam scores across grade levels', 'link': '/education/line'}
                             ],
                             code_example='''import matplotlib.pyplot as plt
import pandas as pd

# Create a simple line plot
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['value'], marker='o', linewidth=2)
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()''')
    
    elif library == 'seaborn':
        return render_template('lesson.html',
                             library_name='Seaborn',
                             overview='Seaborn is a statistical data visualization library built on top of matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics with beautiful default styles and color palettes.',
                             features=[
                                 {'title': 'Statistical Plots', 'description': 'Built-in support for statistical visualizations like distributions, correlations, and regressions.'},
                                 {'title': 'Beautiful Defaults', 'description': 'Attractive default styles and color palettes that make plots publication-ready with minimal code.'},
                                 {'title': 'Easy Integration', 'description': 'Works seamlessly with Pandas DataFrames, making data exploration quick and intuitive.'},
                                 {'title': 'Advanced Visualizations', 'description': 'Specialized plots like heatmaps, violin plots, and pair plots for complex data analysis.'}
                             ],
                             best_for=[
                                 'Statistical data exploration',
                                 'Distribution analysis',
                                 'Correlation and relationship discovery',
                                 'Categorical data comparison'
                             ],
                             domain_uses=[
                                 'Health: BMI distributions, diagnosis comparisons, correlation heatmaps',
                                 'E-Commerce: Category sales comparisons, order value distributions, country-device heatmaps',
                                 'Education: Subject score comparisons, program effectiveness analysis',
                                 'Finance: Interest rate distributions, risk analysis scatter plots, correlation matrices'
                             ],
                             examples=[
                                 {'name': 'Category Sales Comparison', 'domain': 'E-Commerce', 'description': 'Bar plot comparing sales across product categories', 'link': '/ecommerce/bar'},
                                 {'name': 'BMI Distribution', 'domain': 'Health', 'description': 'Histogram with KDE showing BMI distribution', 'link': '/health/histogram'},
                                 {'name': 'Health Metrics Correlation', 'domain': 'Health', 'description': 'Heatmap showing correlations between health metrics', 'link': '/health/heatmap'},
                                 {'name': 'Risk Analysis Scatter', 'domain': 'Finance', 'description': 'Scatter plot analyzing loan risk factors', 'link': '/finance/scatter'}
                             ],
                             code_example='''import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")

# Create a bar plot
sns.barplot(x='category', y='sales', data=df, palette='viridis')
plt.title('Sales by Category')
plt.show()

# Create a histogram with KDE
sns.histplot(data=df, x='bmi', kde=True, bins=10)
plt.show()

# Create a heatmap
correlation = df[['col1', 'col2', 'col3']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()''')
    
    elif library == 'bokeh':
        return render_template('lesson.html',
                             library_name='Bokeh',
                             overview='Bokeh is an interactive visualization library that targets modern web browsers for presentation. It provides elegant, concise construction of versatile graphics and gives high-performance interactivity over large or streaming datasets.',
                             features=[
                                 {'title': 'Interactive Plots', 'description': 'Create interactive visualizations with zooming, panning, and hover tooltips that work in web browsers.'},
                                 {'title': 'Web-Based', 'description': 'Designed for web deployment, making it easy to embed interactive plots in web applications.'},
                                 {'title': 'Real-Time Data', 'description': 'Excellent for streaming data and real-time updates in dashboards.'},
                                 {'title': 'Custom Interactions', 'description': 'Build custom interactions and widgets for complex data exploration.'}
                             ],
                             best_for=[
                                 'Interactive web dashboards',
                                 'Real-time data visualization',
                                 'Data exploration tools',
                                 'Embedded web applications'
                             ],
                             domain_uses=[
                                 'Education: Interactive student performance dashboards',
                                 'E-Commerce: Real-time sales monitoring',
                                 'Finance: Live loan portfolio analysis',
                                 'Health: Interactive patient data exploration'
                             ],
                             examples=[
                                 {'name': 'Interactive Student Dashboard', 'domain': 'Education', 'description': 'Interactive scatter plot with tooltips for student performance data', 'link': '/education/interactive'}
                             ],
                             code_example='''from bokeh.plotting import figure, show
from bokeh.embed import components

# Create a figure
p = figure(title="Interactive Plot", 
           x_axis_label='X Axis', 
           y_axis_label='Y Axis',
           width=800, height=600)

# Add data
p.circle(x_data, y_data, size=10, color='blue', alpha=0.7)

# Add hover tooltips
p.add_tools(HoverTool(tooltips=[("X", "@x"), ("Y", "@y")]))

# Embed in web page
script, div = components(p)
# Use script and div in HTML template''')
    
    elif library == 'plotly':
        return render_template('lesson.html',
                             library_name='Plotly',
                             overview='Plotly is a comprehensive graphing library that makes interactive, publication-quality graphs. It supports a wide range of chart types and can be used in Python, R, and JavaScript. Plotly graphs are interactive by default and can be easily embedded in web applications.',
                             features=[
                                 {'title': 'Interactive Dashboards', 'description': 'Create fully interactive dashboards with zoom, pan, hover, and click interactions.'},
                                 {'title': '3D Visualizations', 'description': 'Support for 3D plots, surface plots, and complex multi-dimensional visualizations.'},
                                 {'title': 'Export Options', 'description': 'Export plots as static images (PNG, SVG) or interactive HTML files.'},
                                 {'title': 'Rich Annotations', 'description': 'Easy-to-add annotations, shapes, and text for enhanced data storytelling.'}
                             ],
                             best_for=[
                                 'Interactive dashboards and reports',
                                 '3D and complex visualizations',
                                 'Data exploration and analysis',
                                 'Web-based data presentations'
                             ],
                             domain_uses=[
                                 'E-Commerce: Interactive revenue dashboards with filtering',
                                 'Health: Interactive patient data exploration',
                                 'Finance: Dynamic loan portfolio analysis',
                                 'Education: Interactive student performance dashboards'
                             ],
                             examples=[
                                 {'name': 'Interactive Revenue Dashboard', 'domain': 'E-Commerce', 'description': 'Interactive scatter plot with hover details for revenue analysis', 'link': '/ecommerce/interactive'},
                                 {'name': 'Interactive Patient Dashboard', 'domain': 'Health', 'description': 'Interactive visualization for exploring patient health metrics', 'link': '/health/interactive'},
                                 {'name': 'Interactive Loan Dashboard', 'domain': 'Finance', 'description': 'Interactive scatter plot for loan risk analysis', 'link': '/finance/interactive'}
                             ],
                             code_example='''import plotly.express as px
import plotly.graph_objects as go

# Simple interactive scatter plot
fig = px.scatter(df, x='x_column', y='y_column', 
                 color='category', size='value',
                 hover_data=['additional_info'],
                 title='Interactive Plot')
fig.show()

# Or create with graph_objects for more control
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_data, y=y_data, 
                        mode='markers',
                        marker=dict(size=10, color=colors)))
fig.update_layout(title='Custom Interactive Plot')
fig.show()''')
    
    else:
        return "Library not found", 404

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5003))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host=host, port=port, debug=debug)

