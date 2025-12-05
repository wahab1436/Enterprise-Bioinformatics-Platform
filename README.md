# Bioinformatics-Unsupervised-Analytics-Platform (Dash)

A **production-grade Dash application** designed for advanced bioinformatics and high-dimensional biological data analysis.  
The platform supports multi-format file uploads (CSV, TSV, XLSX, JSON, Parquet, Feather), automatic preprocessing, **unsupervised machine learning**, PCA/UMAP embeddings, anomaly detection, and complete export capabilities — all inside an interactive Dash interface.

---

## Project Overview

This platform is built for **researchers, data scientists, and bioinformatics engineers** who work with large-scale gene expression, transcriptomics, proteomics, and multi-omics datasets.

It provides a complete workflow for **unsupervised machine learning in bioinformatics**, including:

- **Automatic dataset ingestion** with robust type and delimiter detection  
- **Multi-stage preprocessing pipeline**: missing value handling, duplicate removal, encoding, scaling  
- **Unsupervised ML algorithms**: K-Means (with auto-k), DBSCAN (with auto-eps), hierarchical clustering  
- **Dimensionality reduction**: PCA and UMAP for high-dimensional visualization  
- **Anomaly detection** using Isolation Forest  
- **Insight generation**: cluster-wise summaries, feature trends, embedding interpretation  
- **Interactive data exploration** through Dash + Plotly components  
- **Export system**: download processed data, embeddings, cluster labels, insights, logs  
- **Session-safe pipeline execution** to avoid crashes on messy or corrupted datasets  

The app is completely modular, scalable, and suitable for research labs, enterprise workflows, and portfolio demonstration.

---

## Features

### 1. Multi-Format File Upload System
Supports:

- CSV  
- TSV  
- XLS / XLSX  
- JSON  
- Parquet  
- Feather  

Includes:

- Automatic delimiter detection  
- Automatic encoding detection  
- Schema inference  
- Robust JSON normalization  
- Corrupted file recovery attempts  

---

### 2. Adaptive Preprocessing Pipeline
The platform performs automated preprocessing steps including:

- Detection and imputation of missing values  
- Removal of duplicates  
- Numeric + categorical feature handling  
- Scaling (StandardScaler, RobustScaler)  
- Outlier removal (optional)  
- Type casting and cleaning  

---

### 3. Unsupervised Machine Learning

#### K-Means Clustering  
- Automatic k selection using inertia + elbow heuristic  
- Cluster labeling  
- Cluster-level statistics  

#### DBSCAN  
- Automatic epsilon detection using the k-distance method  
- Noise/outlier handling  

#### Hierarchical Clustering (Optional)  
- Agglomerative clustering  
- Distance thresholding  

---

### 4. Dimensionality Reduction

#### PCA (Principal Component Analysis)  
- Variance explained  
- Contribution of components  
- 2D embeddings  

#### UMAP  
- Non-linear embedding  
- Configurable neighbors and minimum distance  

Both embeddings are used for interactive cluster visualization.

---

### 5. Anomaly Detection

Uses **Isolation Forest** to detect:

- Sample-level anomalies  
- Expression outliers  
- Unusual biological profiles  

Provides:

- Anomaly scores  
- Outlier labels  
- Visual exploration  

---

### 6. Interactive Visualizations

All visualizations are rendered using Plotly and are fully interactive:

- PCA scatter plots  
- UMAP scatter plots  
- Clustered 2D projections  
- Heatmaps  
- Pair distributions  
- Correlation matrices  
- Feature histograms  
- Outlier distribution plots  

Each visualization supports zoom, hover, lasso selection, and export as PNG.

---

### 7. Export System

You can export:

- Cleaned dataset  
- Cluster labels  
- PCA / UMAP embeddings  
- Anomaly scores  
- Summary statistics  
- Logs and reports  

Export formats:

- CSV  
- XLSX  
- JSON  
- ZIP (multi-file export)  

---

## Installation

### 1. Clone the repository  
```bash
git clone https://(https://github.com/wahab1436/Enterprise-Bioinformatics-Platform)
2. Install the required dependencies
bash
Copy code
pip install -r requirements.txt



How the Pipeline Works
1. File Upload
User uploads a dataset → backend detects:

File type

Delimiter

Encoding

Columns & dtypes

2. Preprocessing
Pipeline transforms the dataset via:

Cleaning

Imputation

Scaling

Feature filtering

Error catching

3. Embedding + Clustering
UMAP/PCA are computed → K-Means/DBSCAN cluster the embeddings.

4. Analysis + Visualizations
Interactive dashboards show:

Cluster separation

Anomaly distribution

Feature correlations

5. Export
Users can download all results.

Use Cases
Gene expression analysis

Single-cell feature clustering

Biomarker discovery

Multi-omics exploratory analysis

Clinical cohort clustering

Patient stratification

Requirements
All required libraries are included in requirements.txt, including:

dash

flask

pandas

numpy

scikit-learn

umap-learn

pyarrow

scipy

kneed

plotly

xlsxwriter

License
This project is released under the MIT License.
Feel free to modify and use it for research or commercial projects.

Author
Abdul Wahab
data scientist
Islamabad, Pakistan


