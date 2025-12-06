🚀 Cloud-Based Business Intelligence & Sales Forecasting Platform (AWS + Streamlit)

A fully automated, cloud-native Business Intelligence (BI) and Sales Forecasting system built using AWS serverless services, machine learning models, and an interactive Streamlit dashboard.
This platform transforms raw user-uploaded datasets into KPIs, visual insights, demand forecasts, and inventory recommendations—without any manual intervention.

📌 Project Overview

This project provides an end-to-end data analytics pipeline:

User uploads a CSV or PDF through the Streamlit UI.

The file is stored in an Amazon S3 Data Lake.

An S3 Event triggers AWS Lambda to validate and route the file.

AWS Glue ETL cleans, formats, and converts the data into optimized Parquet files.

Amazon Athena queries the curated data for BI insights.

ARIMA / Prophet / SageMaker models generate demand forecasts.

Inventory optimization algorithms compute EOQ, reorder levels, and detect dead stock.

A Streamlit dashboard displays KPIs, charts, forecasts, and downloadable reports.

The system is fully serverless, scalable, cost-efficient, and ideal for retail, supply-chain, and sales analytics use cases.

🏗 Architecture
User → Streamlit UI → S3 (Raw Zone) → Lambda → Glue (ETL) → S3 (Curated) 
     → Athena → ML Models (Prophet/ARIMA/SageMaker) → S3 (Predictions)
     → Streamlit Dashboard (KPIs, Visuals, Inventory Insights)


Includes Data Lake layers:

raw

curated

training

predictions

logs

scripts

athena-results

✨ Key Features
🔹 1. Automated Data Ingestion

Drag-and-drop file upload

Accepts CSV and PDF (table extraction)

Stored securely in S3

🔹 2. ETL & Data Transformation (AWS Glue)

Cleans numeric/date fields

Removes duplicates & anomalies

Converts to optimized Parquet format

Auto-schema detection

🔹 3. BI Analytics (KPIs & Dashboards)

Total revenue, profit, top products

Regional sales breakdown

Customer segmentation (RFM)

Time-series trend charts

Product-level performance reports

🔹 4. Demand Forecasting (ML Models)

Supports:

Prophet

ARIMA

Amazon SageMaker models

Outputs:

Forecast values

Confidence intervals

Seasonal & trend components

🔹 5. Inventory Optimization

Economic Order Quantity (EOQ)

Reorder Point (ROP)

Safety Stock calculations

Dead Stock identification (no-sale products)

🔹 6. Streamlit Dashboard

Interactive plots (Plotly)

Downloadable reports

Auto-updated S3-stored outputs

🛠 Tech Stack
Frontend & Visualization

Streamlit

Plotly

Python (Pandas, NumPy)

Backend Services

AWS S3

AWS Lambda

AWS Glue

AWS Athena

AWS CloudWatch

AWS SNS

Machine Learning

Prophet

Statsmodels (ARIMA)

Amazon SageMaker

Other Tools

Pandas / NumPy

PyPDF2 (PDF table extraction)

FastParquet / PyArrow

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

🚀 How to Run Locally
1. Clone the repository
git clone https://github.com/your-username/cloud-bi-sales-forecasting.git
cd cloud-bi-sales-forecasting

2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run Streamlit Dashboard
streamlit run streamlit_app.py

5. Upload data and start analysis

Navigate to:

http://localhost:8501

☁️ Deploying on AWS

The project includes Terraform modules for:

S3 bucket creation

Lambda deployment

Glue Crawler & Job setup

SageMaker model configuration

IAM roles and permissions

Deploy with:

terraform init
terraform apply

📊 Sample Outputs

Sales KPIs

Product revenue charts

Regional heat maps

Forecast vs actual plots

EOQ optimization tables

Dead stock list

📘 Future Enhancements

Real-time streaming ingestion (Kinesis)

Multi-user authentication (Cognito)

Deep-learning forecasting models

Deployment via Docker + ECS / EKS

Multi-store consolidated analytics

🤝 Contributing

Contributions, issues, and feature requests are welcome!

Fork the repo

Create a new branch

Commit changes

Open a PR

📄 License

This project is licensed under the MIT License.
