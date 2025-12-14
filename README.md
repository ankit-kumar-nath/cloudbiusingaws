# Cloud BI - A Cloud-Based Business Intelligence & Sales Forecasting System Solution

A scalable, cloud-native **Business Intelligence (BI)** and **Sales Forecasting** platform built using **AWS serverless services**, **machine learning models**, and an **interactive Streamlit dashboard**.  
The system automates the entire analytics lifecycle—from raw data ingestion to forecasting and inventory optimization.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [AWS Deployment](#aws-deployment)
- [Sample Outputs](#sample-outputs)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Traditional BI tools mainly focus on visualization and require manual data preparation.  
This project delivers a **fully automated BI pipeline** that ingests user-uploaded sales data, processes it using cloud ETL, applies forecasting models, and presents actionable insights through a dashboard.

Designed for **retail, e-commerce, supply chain, and sales analytics** use cases.

---

## Problem Statement

Businesses often face:
- Manual analysis using spreadsheets  
- Inaccurate sales forecasting  
- Inefficient inventory planning  
- High licensing costs for BI tools  

There is a need for an **automated, scalable, and cost-effective BI solution**.

---

## Solution Overview

This project provides:
- A **serverless AWS data pipeline**
- A **data lake architecture** for structured analytics
- **Machine learning models** for demand forecasting
- **Inventory optimization logic**
- A **user-friendly Streamlit dashboard**

---

## System Architecture

User Upload → S3 (Raw)
→ Lambda (Validation)
→ Glue (ETL)
→ S3 (Curated / Training)
→ Athena (Analytics)
→ Forecast Models (ARIMA / Prophet / SageMaker)
→ S3 (Predictions)
→ Streamlit Dashboard


### S3 Data Lake Zones
- raw  
- curated  
- training  
- predictions  
- logs  
- scripts  
- athena-results  

---

## Key Features

### Data Ingestion
- Upload CSV or PDF files  
- Automatic schema detection  
- Secure storage in Amazon S3  

### ETL & Processing
- AWS Glue for data cleaning and transformation  
- Parquet conversion for faster queries  
- Partitioned datasets  

### Business Intelligence
- Revenue and profit KPIs  
- Product and regional analysis  
- Customer segmentation (RFM)  
- Interactive visualizations  

### Demand Forecasting
- ARIMA  
- Prophet  
- SageMaker models  
- Confidence intervals  

### Inventory Optimization
- EOQ (Economic Order Quantity)  
- Reorder Point calculation  
- Safety stock estimation  
- Dead stock identification  

### Dashboard
- Streamlit-based interactive UI  
- KPI cards and charts  
- Downloadable reports  

---

## Technology Stack

### Cloud Services
- AWS S3  
- AWS Lambda  
- AWS Glue  
- Amazon Athena  
- Amazon SageMaker  
- AWS CloudWatch  
- AWS IAM  

### Programming & Libraries
- Python  
- Pandas, NumPy  
- Plotly  
- Statsmodels  
- Prophet  
- Streamlit  

---

## Project Structure
```
├── ingestion/
│ ├── csv_loader.py
│ ├── pdf_loader.py
│ ├── schema_detector.py
│
├── profiling/
│ ├── profiler.py
│
├── utils/
│ ├── s3_uploader.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── terraform/
├── s3.tf
├── glue.tf
├── lambda.tf
├── sagemaker.tf

```
---

## Installation & Setup

### Prerequisites
- Python 3.9+
- AWS Account
- Terraform (optional)

### Clone Repository
```bash
git clone https://github.com/your-username/cloud-bi-sales-forecasting.git
cd cloud-bi-sales-forecasting
```
### Clone Repository
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
## Running the Application
```bash
streamlit run streamlit_app.py
```
### Open in browser:
```bash
http://localhost:8501
```
### Check out the Proeject 
```bash
https://cloudbiusingaws.streamlit.app/
```

## AWS Deployment
Infrastructure is provisioned using Terraform, including:
- S3 buckets
- Lambda functions
- Glue ETL jobs
- SageMaker models
- IAM roles
```bash
terraform init
terraform apply
```

## Sample Outputs
- Sales KPI dashboard
- Product-wise revenue charts
- Region-wise heatmaps
- Forecast vs actual plots
- Inventory EOQ & reorder metrics
- Dead stock reports

## Future Enhancements
- Real-time streaming with Kinesis
- Multi-user authentication (Cognito)
- Deep learning models (LSTM, DeepAR)
- Docker / ECS deployment
- Automated model retraining pipelines

## Contributing
Contributions are welcome:

- Fork the repository
- Create a feature branch
- Commit your changes
- Open a Pull Request

## Documentation 
Get it from the following Google Drive-
```bash
https://drive.google.com/file/d/1AN7n8wwdPBQNuS6tiarvYZlXAJflUkPV/view?usp=sharing
```
## License
This project is licensed under the MIT License.
