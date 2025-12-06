Cloud-Based Business Intelligence & Sales Forecasting Platform (AWS + Streamlit)

A fully automated cloud-native BI, Forecasting, and Inventory Optimization System.

📌 Overview

This project implements an end-to-end Business Intelligence system powered by AWS.
Users can upload sales data (CSV/PDF), which is automatically processed through a serverless cloud pipeline to generate:

KPI dashboards

Sales trends & visual analytics

Demand forecasting (ARIMA / Prophet / SageMaker)

Inventory optimization (EOQ, Reorder Point, Dead Stock)

The final insights are displayed on an interactive Streamlit dashboard.

🏗️ System Architecture
User Upload → S3 (Raw Zone)
            → Lambda (Validation)
            → Glue (ETL)
            → S3 (Curated Zone)
            → Athena (Query Engine)
            → Forecast Models (ARIMA / Prophet / SageMaker)
            → S3 (Predictions)
            → Streamlit Dashboard

S3 Data Lake Structure
raw/
curated/
training/
predictions/
logs/
scripts/
athena-results/

✨ Key Features
🔹 1. Automated Ingestion

Upload CSV or PDF

Auto-schema detection

S3 storage with serverless triggers

🔹 2. ETL Processing (AWS Glue)

Cleans and validates data

Removes duplicates & fixes formats

Converts to Parquet for faster queries

🔹 3. BI Analytics

Revenue, profit, product performance

Regional sales analysis

Customer segmentation (RFM)

Time-series visualizations

🔹 4. Demand Forecasting

Supports:

Prophet

ARIMA

SageMaker models

Outputs include:

Forecast values

Confidence intervals

Seasonal components

🔹 5. Inventory Optimization

EOQ (Economic Order Quantity)

Reorder Point calculation

Safety stock estimation

Dead stock identification

🔹 6. Interactive Dashboard (Streamlit)

KPI cards

Forecast charts

Downloadable reports

Clean and user-friendly UI

🛠️ Tech Stack
Frontend / Dashboard

Streamlit

Plotly

Pandas, NumPy

Cloud Services (AWS)

S3

Lambda

Glue

Athena

SageMaker

CloudWatch

IAM

Machine Learning

Prophet

Statsmodels (ARIMA)

Amazon SageMaker

📂 Project Structure
├── ingestion/
│   ├── csv_loader.py
│   ├── pdf_loader.py
│   ├── schema_detector.py
│
├── profiling/
│   ├── profiler.py
│
├── utils/
│   ├── s3_uploader.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── terraform/
    ├── s3.tf
    ├── glue.tf
    ├── lambda.tf
    ├── sagemaker.tf

🚀 Running the Project Locally
1. Clone the repository
git clone https://github.com/your-username/cloud-bi-sales-forecasting.git
cd cloud-bi-sales-forecasting

2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

4. Start the dashboard
streamlit run streamlit_app.py


Open browser at http://localhost:8501
.

☁️ Deploying on AWS

Terraform scripts included for automated deployment of:

S3 bucket

Lambda functions

Glue crawlers & ETL jobs

SageMaker model

IAM roles

Deploy with:

terraform init
terraform apply

📊 Sample Outputs

Sales KPIs

Product performance charts

Regional sales heatmap

Forecast vs actual plot

EOQ & inventory metrics

Dead stock list

🔮 Future Enhancements

Real-time streaming ingestion (Kinesis)

Multi-user authentication (Cognito)

Deep-learning forecasting (LSTM, DeepAR)

Docker / ECS deployment

Automated model retraining (Pipelines)
