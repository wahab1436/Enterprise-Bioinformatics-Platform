# ============================================
# bioinformatics_platform.py
# Enterprise Bioinformatics Unsupervised Analytics Platform
# Version: 4.0.0 - Production Grade
# ============================================

import os
import sys
import json
import base64
import logging
import traceback
import warnings
from datetime import datetime
from pathlib import Path
import io

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context, dash_table
    import dash_bootstrap_components as dbc
    from flask import Flask
    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.feature_selection import VarianceThreshold
    import umap.umap_ as umap
    import scipy.stats as stats
    from scipy import sparse
    
    logger.info("All libraries imported successfully")
    
except ImportError as e:
    logger.error(f"Missing library: {e}")
    print("Install: pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn umap-learn scipy")
    sys.exit(1)

# ============================================
# CONFIGURATION
# ============================================

class PlatformConfig:
    PRIMARY_COLOR = "#6B8F7A"  # Moss green
    SECONDARY_COLOR = "#2E4C3D"
    BACKGROUND_COLOR = "#F8F9FA"
    CARD_BACKGROUND = "#FFFFFF"
    ACCENT_COLOR = "#8A9A5B"
    TEXT_COLOR = "#333333"
    
    ALLOWED_EXTENSIONS = {'csv', 'tsv', 'txt', 'xlsx', 'xls', 'parquet', 'json', 'feather'}
    MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
    
    DEFAULT_CLUSTERS = 4
    PLOT_HEIGHT = 550
    TABLE_HEIGHT = 400
    
    @staticmethod
    def get_color_palette(n_colors):
        if n_colors <= 10:
            return px.colors.qualitative.Set3[:n_colors]
        else:
            return px.colors.sample_colorscale("Viridis", n_colors)

# ============================================
# ADVANCED DATA MANAGER
# ============================================

class AdvancedBioinformaticsDataManager:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.metadata = None
        self.embeddings = {}
        self.clustering_results = {}
        self.anomaly_scores = None
        self.marker_genes = {}
        self.pathway_results = {}
        self.survival_data = None
        self.validation_results = {}
        self.pipeline_log = []
        self.data_insights = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_summary = {}
        
        logger.info(f"Initialized Advanced Data Manager: {self.session_id}")
    
    def log_pipeline_step(self, step, status, message, execution_time=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'status': status,
            'message': message,
            'execution_time': execution_time
        }
        self.pipeline_log.append(log_entry)
        logger.info(f"{step}: {status} - {message}")
    
    def add_data_insight(self, insight_type, message, severity="info"):
        self.data_insights.append({
            'type': insight_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def detect_file_format(self, filename):
        ext = filename.lower().split('.')[-1] if '.' in filename else 'txt'
        return ext if ext in PlatformConfig.ALLOWED_EXTENSIONS else 'txt'
    
    def universal_data_loader(self, content, filename):
        """Load any data format with comprehensive error handling"""
        try:
            start_time = datetime.now()
            
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            file_ext = self.detect_file_format(filename)
            
            df = None
            load_attempts = []
            
            # Try different reading strategies
            if file_ext in ['csv', 'tsv', 'txt']:
                # Strategy 1: Auto-detect delimiter
                for delimiter in [',', '\t', ';', '|', ' ']:
                    try:
                        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), 
                                        delimiter=delimiter, low_memory=False)
                        if df.shape[1] > 1:
                            load_attempts.append(f"Success with delimiter '{delimiter}'")
                            break
                    except Exception as e:
                        load_attempts.append(f"Failed with delimiter '{delimiter}': {str(e)}")
                        continue
                
                # Strategy 2: Python engine
                if df is None or df.shape[1] <= 1:
                    try:
                        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), 
                                        sep=None, engine='python', low_memory=False)
                        load_attempts.append("Success with python engine")
                    except Exception as e:
                        load_attempts.append(f"Python engine failed: {str(e)}")
                
                # Strategy 3: Try different encodings
                if df is None:
                    for encoding in ['latin1', 'iso-8859-1', 'cp1252', 'utf-16']:
                        try:
                            df = pd.read_csv(io.StringIO(decoded.decode(encoding)), 
                                            sep=None, engine='python', low_memory=False)
                            if df is not None and df.shape[1] > 1:
                                load_attempts.append(f"Success with encoding {encoding}")
                                break
                        except:
                            continue
            
            elif file_ext in ['xlsx', 'xls']:
                try:
                    df = pd.read_excel(io.BytesIO(decoded))
                    load_attempts.append("Success reading Excel file")
                except Exception as e:
                    load_attempts.append(f"Excel read failed: {str(e)}")
            
            elif file_ext == 'parquet':
                try:
                    df = pd.read_parquet(io.BytesIO(decoded))
                    load_attempts.append("Success reading Parquet file")
                except Exception as e:
                    load_attempts.append(f"Parquet read failed: {str(e)}")
            
            elif file_ext == 'json':
                try:
                    df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
                    load_attempts.append("Success reading JSON file")
                except Exception as e:
                    try:
                        data = json.loads(decoded.decode('utf-8'))
                        df = pd.json_normalize(data)
                        load_attempts.append("Success normalizing JSON")
                    except Exception as e2:
                        load_attempts.append(f"JSON processing failed: {str(e2)}")
            
            elif file_ext == 'feather':
                try:
                    import pyarrow.feather as feather
                    df = feather.read_feather(io.BytesIO(decoded))
                    load_attempts.append("Success reading Feather file")
                except Exception as e:
                    load_attempts.append(f"Feather read failed: {str(e)}")
            
            # Final fallback: Read as raw text
            if df is None:
                try:
                    lines = decoded.decode('utf-8').split('\n')
                    data = []
                    for line in lines[:100]:  # First 100 lines
                        if line.strip():
                            data.append(line.strip().split())
                    df = pd.DataFrame(data)
                    load_attempts.append("Success reading as raw text")
                except Exception as e:
                    raise ValueError(f"Could not parse file. Attempts: {load_attempts}")
            
            # Clean and validate the dataframe
            df = self.clean_and_validate_dataframe(df)
            
            # Generate data insights
            self.generate_initial_insights(df)
            
            self.raw_data = df
            
            # Create comprehensive data summary
            self.data_summary = {
                'filename': filename,
                'file_type': file_ext,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_values': int(df.isnull().sum().sum()),
                'missing_percentage': float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
                'duplicate_rows': int(df.duplicated().sum()),
                'load_attempts': load_attempts
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.log_pipeline_step(
                'data_loading', 
                'completed', 
                f"Loaded {self.data_summary['rows']} rows, {self.data_summary['columns']} columns",
                execution_time
            )
            
            return True, self.data_summary
            
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.log_pipeline_step('data_loading', 'failed', error_msg)
            return False, error_msg
    
    def clean_and_validate_dataframe(self, df):
        """Advanced data cleaning with validation"""
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.loc[:, df.notna().any()]
        
        # Handle duplicate column names
        if len(df.columns) != len(set(df.columns)):
            new_columns = []
            col_counts = {}
            for col in df.columns:
                if col in col_counts:
                    col_counts[col] += 1
                    new_columns.append(f"{col}_{col_counts[col]}")
                else:
                    col_counts[col] = 0
                    new_columns.append(col)
            df.columns = new_columns
        
        # Convert object columns that contain numeric data
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().mean() > 0.8:  # If >80% successful
                    df[col] = converted
                    self.add_data_insight(
                        'data_type_conversion',
                        f"Column '{col}' converted from text to numeric",
                        'info'
                    )
            except:
                pass
        
        # Convert potential datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().any():
                    self.add_data_insight(
                        'data_type_conversion',
                        f"Column '{col}' identified as datetime",
                        'info'
                    )
            except:
                pass
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Add data quality insights
        self.assess_data_quality(df)
        
        return df
    
    def assess_data_quality(self, df):
        """Assess data quality and generate insights"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        if missing_percentage > 50:
            self.add_data_insight(
                'data_quality',
                f"High missing values detected ({missing_percentage:.1f}%). Consider data imputation.",
                'warning'
            )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.add_data_insight(
                'data_quality',
                "No numeric columns found. Clustering may require numeric features.",
                'warning'
            )
        
        # Check for constant columns
        for col in numeric_cols:
            if df[col].nunique() == 1:
                self.add_data_insight(
                    'data_quality',
                    f"Column '{col}' has constant values. Consider removing.",
                    'info'
                )
    
    def generate_initial_insights(self, df):
        """Generate initial insights about the dataset"""
        self.data_insights = []  # Clear previous insights
        
        # Basic statistics
        self.add_data_insight(
            'dataset_overview',
            f"Dataset contains {df.shape[0]} samples and {df.shape[1]} features",
            'info'
        )
        
        # Data types
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        
        self.add_data_insight(
            'data_types',
            f"Numeric features: {numeric_count}, Categorical features: {categorical_count}",
            'info'
        )
        
        # Missing values
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_pct = missing_total / (df.shape[0] * df.shape[1]) * 100
            self.add_data_insight(
                'missing_values',
                f"Missing values detected: {missing_total} ({missing_pct:.1f}% of data)",
                'warning' if missing_pct > 10 else 'info'
            )
        
        # Data distribution insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for skewed distributions
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                skewness = stats.skew(df[col].dropna())
                if abs(skewness) > 1:
                    self.add_data_insight(
                        'distribution',
                        f"Column '{col}' shows {['negative', 'positive'][skewness > 0]} skewness (value: {skewness:.2f})",
                        'info'
                    )
    
    def adaptive_preprocessing_pipeline(self):
        """Comprehensive adaptive preprocessing pipeline"""
        self.log_pipeline_step('preprocessing', 'started', 'Starting adaptive preprocessing')
        
        try:
            start_time = datetime.now()
            
            if self.raw_data is None:
                raise ValueError("No data available for preprocessing")
            
            df = self.raw_data.copy()
            
            # Track preprocessing steps
            preprocessing_log = []
            
            # Separate data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Step 1: Handle missing values
            missing_before = df.isnull().sum().sum()
            
            if numeric_cols:
                # Remove numeric columns with >50% missing values
                cols_to_keep = []
                for col in numeric_cols:
                    missing_pct = df[col].isnull().mean()
                    if missing_pct < 0.5:
                        cols_to_keep.append(col)
                    else:
                        preprocessing_log.append(f"Removed numeric column '{col}' (missing: {missing_pct:.1%})")
                        self.add_data_insight(
                            'preprocessing',
                            f"Removed column '{col}' due to high missing values ({missing_pct:.1%})",
                            'warning'
                        )
                
                numeric_cols = cols_to_keep
                
                # Impute remaining missing values
                if numeric_cols:
                    # Use KNN imputer for datasets with patterns
                    if len(df) > 100 and len(numeric_cols) > 3:
                        try:
                            imputer = KNNImputer(n_neighbors=5)
                            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                            preprocessing_log.append(f"KNN imputation applied to {len(numeric_cols)} numeric columns")
                        except:
                            # Fallback to median imputation
                            imputer = SimpleImputer(strategy='median')
                            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                            preprocessing_log.append(f"Median imputation applied to {len(numeric_cols)} numeric columns")
                    else:
                        imputer = SimpleImputer(strategy='median')
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        preprocessing_log.append(f"Median imputation applied to {len(numeric_cols)} numeric columns")
            
            # Handle categorical missing values
            if categorical_cols:
                for col in categorical_cols:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna('Missing')
                        preprocessing_log.append(f"Missing values in '{col}' filled with 'Missing'")
            
            missing_after = df.isnull().sum().sum()
            
            # Step 2: Remove duplicates
            duplicates_before = df.duplicated().sum()
            df = df.drop_duplicates()
            duplicates_after = df.duplicated().sum()
            if duplicates_before > duplicates_after:
                preprocessing_log.append(f"Removed {duplicates_before - duplicates_after} duplicate rows")
                self.add_data_insight(
                    'preprocessing',
                    f"Removed {duplicates_before - duplicates_after} duplicate rows",
                    'info'
                )
            
            # Step 3: Handle outliers (optional, based on data size)
            if numeric_cols and len(df) > 100:
                outlier_counts = {}
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        # Cap outliers instead of removing
                        df[col] = df[col].clip(lower_bound, upper_bound)
                        outlier_counts[col] = outliers
                
                if outlier_counts:
                    preprocessing_log.append(f"Capped outliers in {len(outlier_counts)} columns")
                    self.add_data_insight(
                        'preprocessing',
                        f"Capped outliers in {len(outlier_counts)} columns using IQR method",
                        'info'
                    )
            
            # Step 4: Feature selection (remove low variance features)
            if numeric_cols and len(numeric_cols) > 10:
                selector = VarianceThreshold(threshold=0.01)  # Remove features with <1% variance
                try:
                    selector.fit(df[numeric_cols])
                    selected_features = selector.get_support()
                    removed_features = [col for col, selected in zip(numeric_cols, selected_features) if not selected]
                    
                    if removed_features:
                        df = df.drop(columns=removed_features)
                        numeric_cols = [col for col in numeric_cols if col not in removed_features]
                        preprocessing_log.append(f"Removed {len(removed_features)} low-variance features")
                        self.add_data_insight(
                            'preprocessing',
                            f"Removed {len(removed_features)} low-variance features",
                            'info'
                        )
                except:
                    pass  # Skip if variance threshold fails
            
            # Step 5: Normalization
            if numeric_cols:
                # Choose scaler based on data characteristics
                if len(df) > 1000:  # Large dataset, use StandardScaler
                    scaler = StandardScaler()
                    scaler_name = "StandardScaler"
                else:  # Small dataset, use RobustScaler (less sensitive to outliers)
                    scaler = RobustScaler()
                    scaler_name = "RobustScaler"
                
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                preprocessing_log.append(f"Applied {scaler_name} to {len(numeric_cols)} numeric columns")
            
            # Step 6: Encode categorical variables
            if categorical_cols:
                # For high cardinality categorical columns, use hashing
                for col in categorical_cols:
                    n_unique = df[col].nunique()
                    if n_unique > 50:  # High cardinality
                        # Create hash encoding
                        df[f"{col}_hash"] = pd.util.hash_array(df[col].values) % 100
                        df = df.drop(columns=[col])
                        preprocessing_log.append(f"Hashed high-cardinality column '{col}' ({n_unique} unique values)")
                        self.add_data_insight(
                            'preprocessing',
                            f"Applied hashing to high-cardinality column '{col}'",
                            'info'
                        )
                    else:
                        # One-hot encode for low cardinality
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        preprocessing_log.append(f"One-hot encoded column '{col}' ({n_unique} categories)")
            
            self.processed_data = df
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            preprocessing_summary = {
                'rows_processed': df.shape[0],
                'columns_processed': df.shape[1],
                'missing_values_handled': missing_before - missing_after,
                'duplicates_removed': duplicates_before - duplicates_after,
                'numeric_features': len(numeric_cols) if 'numeric_cols' in locals() else 0,
                'preprocessing_steps': preprocessing_log,
                'execution_time': execution_time
            }
            
            self.log_pipeline_step(
                'preprocessing',
                'completed',
                f"Preprocessing completed with {len(preprocessing_log)} steps",
                execution_time
            )
            
            # Generate preprocessing insights
            self.generate_preprocessing_insights(preprocessing_summary)
            
            return True, preprocessing_summary
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.log_pipeline_step('preprocessing', 'failed', error_msg)
            self.add_data_insight('error', f"Preprocessing error: {str(e)}", 'error')
            return False, error_msg
    
    def generate_preprocessing_insights(self, summary):
        """Generate insights about preprocessing results"""
        self.add_data_insight(
            'preprocessing_summary',
            f"Processed {summary['rows_processed']} rows with {summary['columns_processed']} features",
            'success'
        )
        
        if summary['missing_values_handled'] > 0:
            self.add_data_insight(
                'preprocessing_detail',
                f"Handled {summary['missing_values_handled']} missing values",
                'info'
            )
        
        if summary['duplicates_removed'] > 0:
            self.add_data_insight(
                'preprocessing_detail',
                f"Removed {summary['duplicates_removed']} duplicate entries",
                'info'
            )
    
    def perform_dimensionality_reduction(self, method='PCA', n_components=2):
        """Advanced dimensionality reduction with auto-configuration"""
        self.log_pipeline_step('dimensionality_reduction', 'started', f'Starting {method}')
        
        try:
            start_time = datetime.now()
            
            if self.processed_data is None:
                raise ValueError("No processed data available")
            
            # Extract numeric features
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError(f"Need at least 2 numeric columns, found {len(numeric_cols)}")
            
            X = self.processed_data[numeric_cols].values
            
            # Auto-configure parameters
            n_samples, n_features = X.shape
            
            if method == 'PCA':
                # Auto-select components to explain 95% variance or max 50 components
                if n_components == 'auto':
                    pca_temp = PCA(n_components=min(50, n_features))
                    pca_temp.fit(X)
                    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                    n_components = np.argmax(cumulative_variance >= 0.95) + 1
                    n_components = max(2, min(n_components, n_features))
                
                reducer = PCA(n_components=n_components, random_state=42)
                embeddings = reducer.fit_transform(X)
                col_names = [f'PC{i+1}' for i in range(embeddings.shape[1])]
                
                # Calculate explained variance
                variance_explained = {}
                for i, var in enumerate(reducer.explained_variance_ratio_):
                    variance_explained[f'PC{i+1}'] = float(var)
                
                total_variance = sum(reducer.explained_variance_ratio_)
                
                self.add_data_insight(
                    'dimensionality_reduction',
                    f"PCA captured {total_variance:.1%} variance with {n_components} components",
                    'info'
                )
            
            elif method == 'UMAP':
                # Auto-configure UMAP parameters
                n_components = min(n_components, n_features)
                n_neighbors = min(15, n_samples // 10)
                min_dist = 0.1
                
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42
                )
                embeddings = reducer.fit_transform(X)
                col_names = [f'UMAP{i+1}' for i in range(embeddings.shape[1])]
                variance_explained = {}
                
                self.add_data_insight(
                    'dimensionality_reduction',
                    f"UMAP reduction to {n_components} dimensions",
                    'info'
                )
            
            # Store embeddings
            self.embeddings[method] = pd.DataFrame(
                embeddings,
                columns=col_names,
                index=self.processed_data.index
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'method': method,
                'n_components': n_components,
                'variance_explained': variance_explained,
                'embeddings_shape': embeddings.shape,
                'execution_time': execution_time
            }
            
            if method == 'PCA' and variance_explained:
                result['total_variance_explained'] = sum(variance_explained.values())
            
            self.log_pipeline_step(
                'dimensionality_reduction',
                'completed',
                f'{method} completed with {n_components} components',
                execution_time
            )
            
            return True, result
            
        except Exception as e:
            error_msg = f"Dimensionality reduction failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.log_pipeline_step('dimensionality_reduction', 'failed', error_msg)
            self.add_data_insight('error', f"Dimensionality reduction error: {str(e)}", 'error')
            return False, error_msg
    
    def determine_optimal_clusters(self, X, max_clusters=10):
        """Determine optimal number of clusters using multiple methods"""
        if len(X) < 10:
            return 2
        
        max_clusters = min(max_clusters, len(X) // 2)
        
        # Elbow method
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Calculate second derivative (curvature)
        if len(inertias) > 2:
            second_deriv = np.diff(np.diff(inertias))
            elbow_k = np.argmax(np.abs(second_deriv)) + 2
        else:
            elbow_k = 2
        
        # Silhouette score method
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        if silhouette_scores:
            silhouette_k = np.argmax(silhouette_scores) + 2
        else:
            silhouette_k = 2
        
        # Combine methods (weighted)
        optimal_k = int((elbow_k + silhouette_k) / 2)
        optimal_k = max(2, min(optimal_k, max_clusters))
        
        self.add_data_insight(
            'clustering',
            f"Optimal cluster count determined: {optimal_k} (Elbow: {elbow_k}, Silhouette: {silhouette_k})",
            'info'
        )
        
        return optimal_k
    
    def perform_clustering(self, method='kmeans', n_clusters=None):
        """Advanced clustering with multiple algorithm support"""
        self.log_pipeline_step('clustering', 'started', f'Starting {method} clustering')
        
        try:
            start_time = datetime.now()
            
            # Use PCA embeddings for clustering
            if 'PCA' not in self.embeddings:
                success, _ = self.perform_dimensionality_reduction('PCA', 2)
                if not success:
                    raise ValueError("Dimensionality reduction failed")
            
            X = self.embeddings['PCA'].values
            
            # Determine optimal clusters if not specified
            if n_clusters is None:
                n_clusters = self.determine_optimal_clusters(X)
            
            n_clusters = min(max(2, n_clusters), min(15, X.shape[0] // 2))
            
            if method == 'kmeans':
                # Adaptive K-means with multiple initializations
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    init='k-means++'
                )
                labels = kmeans.fit_predict(X)
                centers = kmeans.cluster_centers_
                
                self.add_data_insight(
                    'clustering',
                    f"K-means clustering completed with {n_clusters} clusters",
                    'success'
                )
            
            elif method == 'dbscan':
                # Adaptive DBSCAN with auto epsilon
                from sklearn.neighbors import NearestNeighbors
                
                # Estimate epsilon using k-distance graph
                neigh = NearestNeighbors(n_neighbors=5)
                neigh.fit(X)
                distances, _ = neigh.kneighbors(X)
                distances = np.sort(distances[:, -1])
                
                # Find knee point
                from kneed import KneeLocator
                try:
                    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
                    eps = distances[kneedle.knee] if kneedle.knee else 0.5
                except:
                    eps = 0.5
                
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X)
                centers = None
                
                n_clusters = len(np.unique(labels[labels != -1]))
                
                if n_clusters == 0:
                    raise ValueError("DBSCAN found no clusters. Try adjusting parameters.")
                
                self.add_data_insight(
                    'clustering',
                    f"DBSCAN clustering found {n_clusters} clusters (eps={eps:.2f})",
                    'success' if n_clusters > 1 else 'warning'
                )
            
            elif method == 'hierarchical':
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = hierarchical.fit_predict(X)
                centers = None
                
                self.add_data_insight(
                    'clustering',
                    f"Hierarchical clustering completed with {n_clusters} clusters",
                    'success'
                )
            
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Calculate metrics
            unique_labels = np.unique(labels[labels != -1]) if method == 'dbscan' else np.unique(labels)
            n_clusters_detected = len(unique_labels)
            
            if n_clusters_detected > 1:
                if method == 'dbscan':
                    valid_mask = labels != -1
                    if np.sum(valid_mask) > 1:
                        silhouette_avg = silhouette_score(X[valid_mask], labels[valid_mask])
                    else:
                        silhouette_avg = 0
                else:
                    silhouette_avg = silhouette_score(X, labels)
            else:
                silhouette_avg = 0
            
            # Calculate cluster statistics
            cluster_stats = {}
            for label in unique_labels:
                mask = labels == label
                size = np.sum(mask)
                if size > 0:
                    cluster_data = X[mask]
                    centroid = np.mean(cluster_data, axis=0)
                    spread = np.std(cluster_data, axis=0).mean()
                    cluster_stats[label] = {
                        'size': size,
                        'percentage': (size / len(labels)) * 100,
                        'centroid': centroid.tolist(),
                        'spread': float(spread)
                    }
            
            # Store results
            self.clustering_results[method] = {
                'labels': labels,
                'centers': centers,
                'n_clusters': n_clusters_detected,
                'silhouette_score': silhouette_avg,
                'method': method,
                'cluster_stats': cluster_stats
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'method': method,
                'n_clusters': n_clusters_detected,
                'silhouette_score': silhouette_avg,
                'labels_assigned': len(labels),
                'execution_time': execution_time,
                'cluster_sizes': [stats['size'] for stats in cluster_stats.values()]
            }
            
            self.log_pipeline_step(
                'clustering',
                'completed',
                f'{method.capitalize()} clustering completed with {n_clusters_detected} clusters',
                execution_time
            )
            
            # Generate clustering insights
            self.generate_clustering_insights(result)
            
            return True, result
            
        except Exception as e:
            error_msg = f"Clustering failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.log_pipeline_step('clustering', 'failed', error_msg)
            self.add_data_insight('error', f"Clustering error: {str(e)}", 'error')
            return False, error_msg
    
    def generate_clustering_insights(self, result):
        """Generate insights about clustering results"""
        if result['n_clusters'] > 1:
            self.add_data_insight(
                'clustering_result',
                f"Identified {result['n_clusters']} distinct clusters with silhouette score {result['silhouette_score']:.3f}",
                'success'
            )
            
            # Check for cluster balance
            sizes = result['cluster_sizes']
            if sizes:
                balance_ratio = min(sizes) / max(sizes)
                if balance_ratio < 0.2:
                    self.add_data_insight(
                        'clustering_balance',
                        "Cluster sizes are imbalanced. Consider adjusting clustering parameters.",
                        'warning'
                    )
        else:
            self.add_data_insight(
                'clustering_result',
                "Only one cluster identified. Data may be homogeneous or need different parameters.",
                'warning'
            )
    
    def detect_anomalies(self, contamination=0.1):
        """Advanced anomaly detection with auto-configuration"""
        self.log_pipeline_step('anomaly_detection', 'started', 'Starting anomaly detection')
        
        try:
            start_time = datetime.now()
            
            if self.processed_data is None:
                raise ValueError("No processed data available")
            
            # Use numeric features for anomaly detection
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for anomaly detection")
            
            X = self.processed_data[numeric_cols].values
            
            # Auto-configure contamination if not specified
            if contamination == 'auto':
                # Estimate contamination based on data characteristics
                if len(X) < 100:
                    contamination = 0.1
                else:
                    # Use median absolute deviation to estimate outliers
                    from scipy.stats import median_abs_deviation
                    mad = median_abs_deviation(X, axis=0)
                    outliers = np.any(np.abs(X - np.median(X, axis=0)) > 3 * mad, axis=1)
                    contamination = min(0.2, max(0.01, np.mean(outliers)))
            
            # Use Isolation Forest with auto-configuration
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            
            anomaly_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.decision_function(X)
            
            # Convert to anomaly flag
            is_anomaly = (anomaly_labels == -1)
            
            self.anomaly_scores = pd.DataFrame({
                'anomaly_score': anomaly_scores,
                'is_anomaly': is_anomaly,
                'anomaly_percentile': stats.percentileofscore(anomaly_scores, anomaly_scores)
            }, index=self.processed_data.index)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            anomaly_summary = {
                'total_samples': len(X),
                'anomalies_detected': int(is_anomaly.sum()),
                'anomaly_percentage': float((is_anomaly.sum() / len(X)) * 100),
                'contamination_used': contamination,
                'execution_time': execution_time
            }
            
            self.log_pipeline_step(
                'anomaly_detection',
                'completed',
                f"Detected {anomaly_summary['anomalies_detected']} anomalies",
                execution_time
            )
            
            # Generate anomaly insights
            self.generate_anomaly_insights(anomaly_summary)
            
            return True, anomaly_summary
            
        except Exception as e:
            error_msg = f"Anomaly detection failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.log_pipeline_step('anomaly_detection', 'failed', error_msg)
            self.add_data_insight('error', f"Anomaly detection error: {str(e)}", 'error')
            return False, error_msg
    
    def generate_anomaly_insights(self, summary):
        """Generate insights about anomaly detection"""
        if summary['anomalies_detected'] > 0:
            self.add_data_insight(
                'anomaly_result',
                f"Detected {summary['anomalies_detected']} anomalies ({summary['anomaly_percentage']:.1f}% of data)",
                'warning' if summary['anomaly_percentage'] > 10 else 'info'
            )
            
            if summary['anomaly_percentage'] > 20:
                self.add_data_insight(
                    'anomaly_high',
                    "High anomaly rate detected. Consider reviewing data quality or adjusting detection sensitivity.",
                    'warning'
                )
        else:
            self.add_data_insight(
                'anomaly_result',
                "No anomalies detected in the dataset",
                'info'
            )
    
    def generate_comprehensive_insights(self):
        """Generate comprehensive AI-driven insights"""
        insights = []
        
        # Data quality insights
        if self.data_summary:
            insights.append({
                'category': 'Data Quality',
                'title': 'Dataset Overview',
                'content': f"The dataset contains {self.data_summary['rows']:,} samples and {self.data_summary['columns']:,} features. "
                          f"{self.data_summary['numeric_columns']} numeric features and {self.data_summary['categorical_columns']} categorical features identified."
            })
            
            if self.data_summary['missing_values'] > 0:
                insights.append({
                    'category': 'Data Quality',
                    'title': 'Missing Values',
                    'content': f"Found {self.data_summary['missing_values']:,} missing values ({self.data_summary['missing_percentage']:.1f}% of total). "
                              "These have been handled through adaptive imputation strategies."
                })
        
        # Clustering insights
        for method, result in self.clustering_results.items():
            if result['n_clusters'] > 1:
                insights.append({
                    'category': 'Clustering Analysis',
                    'title': f'{method.upper()} Clustering Results',
                    'content': f"Identified {result['n_clusters']} distinct clusters with a silhouette score of {result['silhouette_score']:.3f}. "
                              f"The clustering suggests {result['n_clusters']} natural groupings in your data."
                })
                
                # Cluster distribution insights
                if 'cluster_stats' in result:
                    sizes = [stats['size'] for stats in result['cluster_stats'].values()]
                    if sizes:
                        largest = max(sizes)
                        smallest = min(sizes)
                        balance = smallest / largest
                        
                        if balance < 0.3:
                            insights.append({
                                'category': 'Clustering Analysis',
                                'title': 'Cluster Balance',
                                'content': f"Clusters show significant size variation (largest: {largest:,}, smallest: {smallest:,}). "
                                          "Consider reviewing clustering parameters or exploring hierarchical relationships."
                            })
        
        # Anomaly insights
        if self.anomaly_scores is not None:
            anomaly_count = self.anomaly_scores['is_anomaly'].sum()
            if anomaly_count > 0:
                insights.append({
                    'category': 'Anomaly Detection',
                    'title': 'Anomaly Overview',
                    'content': f"Detected {anomaly_count:,} anomalous samples using Isolation Forest. "
                              "These samples exhibit characteristics significantly different from the majority."
                })
        
        # Dimensionality reduction insights
        if 'PCA' in self.embeddings:
            insights.append({
                'category': 'Dimensionality Reduction',
                'title': 'PCA Analysis',
                'content': "Principal Component Analysis has been applied to reduce feature dimensionality "
                          "while preserving maximum variance for clustering and visualization."
            })
        
        # Recommendations
        recommendations = []
        
        if self.data_summary and self.data_summary['missing_values'] > 1000:
            recommendations.append("Consider reviewing the source of missing values for data collection improvements.")
        
        if self.clustering_results:
            recommendations.append("Explore different clustering algorithms to compare results and validate cluster stability.")
        
        if self.anomaly_scores is not None and self.anomaly_scores['is_anomaly'].sum() > 0:
            recommendations.append("Investigate anomalous samples for potential data quality issues or interesting edge cases.")
        
        if recommendations:
            insights.append({
                'category': 'Recommendations',
                'title': 'Next Steps',
                'content': " • " + "\n • ".join(recommendations)
            })
        
        return insights
    
    def export_results(self, export_format='csv', include_data=True, include_clusters=True, 
                      include_anomalies=True, include_embeddings=False):
        """Export analysis results in various formats"""
        try:
            export_data = {}
            
            if include_data and self.processed_data is not None:
                export_data['processed_data'] = self.processed_data
            
            if include_clusters and self.clustering_results:
                cluster_data = {}
                for method, result in self.clustering_results.items():
                    cluster_data[f'{method}_cluster'] = result['labels']
                export_data['clustering_results'] = pd.DataFrame(cluster_data)
            
            if include_anomalies and self.anomaly_scores is not None:
                export_data['anomaly_scores'] = self.anomaly_scores
            
            if include_embeddings and self.embeddings:
                for method, embedding in self.embeddings.items():
                    export_data[f'{method}_embeddings'] = embedding
            
            # Add insights summary
            insights_df = pd.DataFrame(self.data_insights)
            export_data['analysis_insights'] = insights_df
            
            # Create export based on format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == 'excel':
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    for sheet_name, df in export_data.items():
                        df.to_excel(writer, sheet_name=sheet_name[:31])
                buffer.seek(0)
                return buffer.getvalue(), f'bioinformatics_analysis_{timestamp}.xlsx'
            
            elif export_format == 'csv':
                # Create a ZIP file with multiple CSVs
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for name, df in export_data.items():
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer)
                        zip_file.writestr(f'{name}.csv', csv_buffer.getvalue())
                
                zip_buffer.seek(0)
                return zip_buffer.getvalue(), f'bioinformatics_analysis_{timestamp}.zip'
            
            elif export_format == 'json':
                export_dict = {}
                for name, df in export_data.items():
                    export_dict[name] = df.to_dict(orient='records')
                
                json_str = json.dumps(export_dict, indent=2)
                return json_str.encode(), f'bioinformatics_analysis_{timestamp}.json'
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def clear_session(self):
        """Clear all session data"""
        self.raw_data = None
        self.processed_data = None
        self.metadata = None
        self.embeddings.clear()
        self.clustering_results.clear()
        self.anomaly_scores = None
        self.marker_genes.clear()
        self.pathway_results.clear()
        self.survival_data = None
        self.validation_results.clear()
        self.pipeline_log.clear()
        self.data_insights.clear()
        self.data_summary = {}
        
        logger.info("Session data cleared")

# ============================================
# DASH APPLICATION
# ============================================

server = Flask(__name__)
server.secret_key = os.environ.get('SECRET_KEY', 'bioinformatics-enterprise-2024')

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Enterprise Bioinformatics Analytics Platform"

data_manager = AdvancedBioinformaticsDataManager()

# ============================================
# CUSTOM STYLES
# ============================================

CUSTOM_STYLES = {
    'background': {
        'backgroundColor': PlatformConfig.BACKGROUND_COLOR,
        'backgroundImage': 'linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%)',
        'minHeight': '100vh'
    },
    'card': {
        'backgroundColor': PlatformConfig.CARD_BACKGROUND,
        'border': '1px solid #e0e0e0',
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)',
        'transition': 'box-shadow 0.3s ease'
    },
    'card_hover': {
        'boxShadow': '0 8px 15px rgba(0, 0, 0, 0.1)'
    },
    'primary_button': {
        'backgroundColor': PlatformConfig.PRIMARY_COLOR,
        'border': 'none',
        'borderRadius': '8px',
        'padding': '10px 24px',
        'fontWeight': '600',
        'color': 'white',
        'transition': 'all 0.3s ease'
    },
    'secondary_button': {
        'backgroundColor': PlatformConfig.SECONDARY_COLOR,
        'border': 'none',
        'borderRadius': '8px',
        'padding': '8px 20px',
        'fontWeight': '500',
        'color': 'white'
    },
    'header': {
        'color': PlatformConfig.SECONDARY_COLOR,
        'fontWeight': '700',
        'fontFamily': "'Inter', sans-serif"
    },
    'subheader': {
        'color': PlatformConfig.TEXT_COLOR,
        'fontWeight': '600',
        'fontFamily': "'Inter', sans-serif"
    },
    'text': {
        'color': PlatformConfig.TEXT_COLOR,
        'fontFamily': "'Inter', sans-serif",
        'lineHeight': '1.6'
    }
}

# ============================================
# UI COMPONENTS
# ============================================

def create_navigation():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.I(className="fas fa-dna", style={
                                        'color': PlatformConfig.PRIMARY_COLOR,
                                        'fontSize': '28px',
                                        'marginRight': '12px'
                                    }),
                                    html.Span(
                                        "Enterprise Bioinformatics Platform",
                                        style={
                                            'fontWeight': '700',
                                            'color': PlatformConfig.SECONDARY_COLOR,
                                            'fontSize': '20px',
                                            'letterSpacing': '-0.5px'
                                        }
                                    )
                                ],
                                className="d-flex align-items-center"
                            ),
                            width="auto"
                        ),
                    ],
                    align="center",
                    className="g-0"
                ),
                
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Span("Data Upload", style={'fontWeight': '500'}),
                                    href="/",
                                    id="nav-upload",
                                    className="nav-link-custom"
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Span("Data Quality", style={'fontWeight': '500'}),
                                    href="/quality",
                                    id="nav-quality",
                                    className="nav-link-custom"
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Span("Clustering", style={'fontWeight': '500'}),
                                    href="/clustering",
                                    id="nav-clustering",
                                    className="nav-link-custom"
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Span("Anomaly Detection", style={'fontWeight': '500'}),
                                    href="/anomaly",
                                    id="nav-anomaly",
                                    className="nav-link-custom"
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Span("Insights", style={'fontWeight': '500'}),
                                    href="/insights",
                                    id="nav-insights",
                                    className="nav-link-custom"
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Span("Export", style={'fontWeight': '500'}),
                                    href="/export",
                                    id="nav-export",
                                    className="nav-link-custom"
                                )
                            ),
                        ],
                        className="ms-auto",
                        navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
            style={'maxWidth': '1400px'}
        ),
        color="white",
        dark=False,
        className="mb-4 border-bottom py-3",
        fixed="top",
        style={'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.08)'}
    )

def create_footer():
    return html.Footer(
        dbc.Container(
            [
                html.Hr(style={'borderColor': '#e0e0e0', 'margin': '40px 0'}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Enterprise Bioinformatics Analytics Platform",
                                            style={
                                                'fontWeight': '600',
                                                'color': PlatformConfig.SECONDARY_COLOR,
                                                'fontSize': '14px'
                                            }
                                        ),
                                        html.Br(),
                                        html.Span(
                                            "Version 4.0 | Professional Edition",
                                            style={
                                                'color': '#666',
                                                'fontSize': '12px',
                                                'marginTop': '4px',
                                                'display': 'block'
                                            }
                                        ),
                                        html.Span(
                                            "© 2024 Bioinformatics Analytics. All rights reserved.",
                                            style={
                                                'color': '#999',
                                                'fontSize': '11px',
                                                'marginTop': '8px',
                                                'display': 'block'
                                            }
                                        )
                                    ]
                                )
                            ],
                            width=6
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.A(
                                            "Documentation",
                                            href="#",
                                            style={
                                                'color': PlatformConfig.PRIMARY_COLOR,
                                                'textDecoration': 'none',
                                                'fontSize': '12px',
                                                'marginRight': '20px'
                                            }
                                        ),
                                        html.A(
                                            "Privacy Policy",
                                            href="#",
                                            style={
                                                'color': PlatformConfig.PRIMARY_COLOR,
                                                'textDecoration': 'none',
                                                'fontSize': '12px',
                                                'marginRight': '20px'
                                            }
                                        ),
                                        html.A(
                                            "Terms of Service",
                                            href="#",
                                            style={
                                                'color': PlatformConfig.PRIMARY_COLOR,
                                                'textDecoration': 'none',
                                                'fontSize': '12px'
                                            }
                                        )
                                    ],
                                    className="text-end"
                                )
                            ],
                            width=6
                        )
                    ],
                    className="align-items-center"
                )
            ],
            fluid=True,
            style={'maxWidth': '1400px'}
        ),
        className="py-4",
        style={'backgroundColor': '#f8f9fa'}
    )

def create_card(title, content, icon=None, color="default"):
    card_styles = {
        'default': {'borderLeft': f'4px solid {PlatformConfig.PRIMARY_COLOR}'},
        'success': {'borderLeft': '4px solid #28a745'},
        'warning': {'borderLeft': '4px solid #ffc107'},
        'info': {'borderLeft': '4px solid #17a2b8'},
        'error': {'borderLeft': '4px solid #dc3545'}
    }
    
    border_style = card_styles.get(color, card_styles['default'])
    
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        html.I(className=f"fas fa-{icon} me-2") if icon else None,
                        html.H5(title, className="mb-0", style=CUSTOM_STYLES['subheader'])
                    ],
                    className="d-flex align-items-center"
                ),
                style={'backgroundColor': 'transparent', 'borderBottom': '1px solid #e0e0e0'}
            ),
            dbc.CardBody(
                content,
                style={'padding': '24px'}
            )
        ],
        style={**CUSTOM_STYLES['card'], **border_style},
        className="mb-4"
    )

def create_metric_card(title, value, change=None, icon=None, color="primary"):
    color_map = {
        'primary': PlatformConfig.PRIMARY_COLOR,
        'secondary': PlatformConfig.SECONDARY_COLOR,
        'success': '#28a745',
        'warning': '#ffc107',
        'info': '#17a2b8'
    }
    
    card_color = color_map.get(color, PlatformConfig.PRIMARY_COLOR)
    
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6(
                                    title,
                                    style={
                                        'color': '#666',
                                        'fontSize': '12px',
                                        'fontWeight': '600',
                                        'textTransform': 'uppercase',
                                        'letterSpacing': '0.5px',
                                        'marginBottom': '8px'
                                    }
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            value,
                                            style={
                                                'fontSize': '28px',
                                                'fontWeight': '700',
                                                'color': card_color,
                                                'lineHeight': '1.2'
                                            }
                                        ),
                                        html.Span(
                                            f" {change}" if change else "",
                                            style={
                                                'fontSize': '12px',
                                                'fontWeight': '500',
                                                'color': '#28a745' if change and '+' in str(change) else '#dc3545',
                                                'marginLeft': '8px'
                                            }
                                        ) if change else None
                                    ],
                                    className="d-flex align-items-baseline"
                                )
                            ],
                            className="flex-grow-1"
                        ),
                        html.Div(
                            html.I(className=f"fas fa-{icon} fa-2x", style={'color': '#e0e0e0'}),
                            className="ms-3"
                        ) if icon else None
                    ],
                    className="d-flex justify-content-between align-items-start"
                )
            ]
        ),
        style={
            **CUSTOM_STYLES['card'],
            'height': '100%',
            'borderLeft': f'4px solid {card_color}'
        }
    )

def create_data_quality_alert(severity, title, message):
    severity_colors = {
        'error': '#dc3545',
        'warning': '#ffc107',
        'info': '#17a2b8',
        'success': '#28a745'
    }
    
    severity_icons = {
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle',
        'success': 'check-circle'
    }
    
    return dbc.Alert(
        [
            html.Div(
                [
                    html.I(
                        className=f"fas fa-{severity_icons.get(severity, 'info-circle')} me-2",
                        style={'fontSize': '18px'}
                    ),
                    html.Strong(title, style={'fontSize': '14px'})
                ],
                className="d-flex align-items-center mb-2"
            ),
            html.P(message, style={'fontSize': '13px', 'marginBottom': '0'})
        ],
        color=severity,
        style={
            'border': f'1px solid {severity_colors.get(severity, "#17a2b8")}',
            'borderLeft': f'4px solid {severity_colors.get(severity, "#17a2b8")}',
            'backgroundColor': f'{severity_colors.get(severity, "#17a2b8")}15',
            'color': '#333'
        },
        className="mb-3"
    )

# ============================================
# PAGE LAYOUTS
# ============================================

def create_upload_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Data Upload & Validation",
                                    style={
                                        **CUSTOM_STYLES['header'],
                                        'fontSize': '32px',
                                        'marginBottom': '12px'
                                    }
                                ),
                                html.P(
                                    "Upload gene expression datasets for comprehensive unsupervised analysis. "
                                    "The platform automatically handles various formats and data quality issues.",
                                    style={
                                        **CUSTOM_STYLES['text'],
                                        'fontSize': '16px',
                                        'color': '#666',
                                        'maxWidth': '800px'
                                    }
                                )
                            ],
                            className="mb-5"
                        )
                    ],
                    width=12
                )
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Upload Dataset",
                            [
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-cloud-upload-alt",
                                                        style={
                                                            'fontSize': '48px',
                                                            'color': PlatformConfig.PRIMARY_COLOR,
                                                            'marginBottom': '20px'
                                                        }
                                                    ),
                                                    html.H5(
                                                        "Drag & Drop or Click to Browse",
                                                        style={
                                                            'fontWeight': '600',
                                                            'marginBottom': '8px'
                                                        }
                                                    ),
                                                    html.P(
                                                        "Supported formats: CSV, TSV, Excel, Parquet, JSON",
                                                        style={'color': '#666', 'fontSize': '14px'}
                                                    ),
                                                    html.P(
                                                        "Maximum file size: 1GB",
                                                        style={'color': '#999', 'fontSize': '12px', 'marginTop': '4px'}
                                                    )
                                                ],
                                                className="text-center"
                                            )
                                        ],
                                        style={
                                            'padding': '60px 40px',
                                            'border': '2px dashed #ddd',
                                            'borderRadius': '10px',
                                            'backgroundColor': '#fafafa',
                                            'cursor': 'pointer',
                                            'transition': 'all 0.3s ease'
                                        },
                                        id='upload-area'
                                    ),
                                    multiple=False,
                                    className="w-100"
                                ),
                                html.Div(id='upload-status', className="mt-4")
                            ],
                            icon="cloud-upload-alt"
                        ),
                        width=8
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Data Requirements",
                            [
                                html.Ul(
                                    [
                                        html.Li(
                                            html.Span(
                                                "First row should contain column headers",
                                                style={'fontSize': '14px'}
                                            )
                                        ),
                                        html.Li(
                                            html.Span(
                                                "Gene expression values should be numeric",
                                                style={'fontSize': '14px'}
                                            )
                                        ),
                                        html.Li(
                                            html.Span(
                                                "Missing values are handled automatically",
                                                style={'fontSize': '14px'}
                                            )
                                        ),
                                        html.Li(
                                            html.Span(
                                                "Clinical metadata can be included",
                                                style={'fontSize': '14px'}
                                            )
                                        ),
                                        html.Li(
                                            html.Span(
                                                "Large datasets up to 1GB are supported",
                                                style={'fontSize': '14px'}
                                            )
                                        )
                                    ],
                                    className="mb-0",
                                    style={'paddingLeft': '20px'}
                                )
                            ],
                            icon="info-circle"
                        ),
                        width=4
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Sample Dataset",
                        [
                            html.P(
                                "For testing purposes, you can load a synthetic bioinformatics dataset:",
                                style={'fontSize': '14px', 'marginBottom': '20px'}
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-vial me-2"),
                                    "Load Sample Dataset"
                                ],
                                id="load-sample-data",
                                color="secondary",
                                className="me-3",
                                style=CUSTOM_STYLES['secondary_button']
                            ),
                            html.Small(
                                "Loads a simulated gene expression dataset for platform evaluation",
                                style={'color': '#999', 'fontSize': '12px'}
                            )
                        ],
                        icon="vial"
                    ),
                    width=12
                )
            )
        ],
        fluid=True,
        style={'paddingTop': '100px', 'maxWidth': '1400px'}
    )

def create_quality_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Data Quality & Preprocessing",
                            style={
                                **CUSTOM_STYLES['header'],
                                'fontSize': '32px',
                                'marginBottom': '12px'
                            }
                        ),
                        html.P(
                            "Comprehensive data quality assessment and adaptive preprocessing",
                            style={
                                **CUSTOM_STYLES['text'],
                                'fontSize': '16px',
                                'color': '#666'
                            }
                        )
                    ],
                    width=12
                ),
                className="mb-5"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Data Quality Metrics",
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(id='quality-metrics'),
                                            width=12
                                        )
                                    ]
                                ),
                                html.Hr(style={'margin': '20px 0'}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(id='data-insights-container'),
                                            width=12
                                        )
                                    ]
                                )
                            ],
                            icon="chart-bar"
                        ),
                        width=8
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Preprocessing Controls",
                            [
                                html.Label("Imputation Strategy", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                dcc.Dropdown(
                                    id='imputation-strategy',
                                    options=[
                                        {'label': 'Median for numeric / Mode for categorical', 'value': 'median_mode'},
                                        {'label': 'K-Nearest Neighbors', 'value': 'knn'},
                                        {'label': 'Forward fill', 'value': 'ffill'},
                                        {'label': 'Remove missing values', 'value': 'remove'}
                                    ],
                                    value='median_mode',
                                    style={'marginBottom': '20px'}
                                ),
                                
                                html.Label("Normalization Method", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                dcc.Dropdown(
                                    id='normalization-method',
                                    options=[
                                        {'label': 'Standard scaling', 'value': 'standard'},
                                        {'label': 'Robust scaling', 'value': 'robust'},
                                        {'label': 'Min-max scaling', 'value': 'minmax'},
                                        {'label': 'No normalization', 'value': 'none'}
                                    ],
                                    value='standard',
                                    style={'marginBottom': '20px'}
                                ),
                                
                                html.Label("Outlier Handling", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                dcc.Dropdown(
                                    id='outlier-method',
                                    options=[
                                        {'label': 'IQR capping', 'value': 'iqr'},
                                        {'label': 'Z-score filtering', 'value': 'zscore'},
                                        {'label': 'No outlier handling', 'value': 'none'}
                                    ],
                                    value='iqr',
                                    style={'marginBottom': '30px'}
                                ),
                                
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-cogs me-2"),
                                        "Run Preprocessing Pipeline"
                                    ],
                                    id="run-preprocessing",
                                    color="primary",
                                    className="w-100",
                                    style=CUSTOM_STYLES['primary_button']
                                )
                            ],
                            icon="sliders-h"
                        ),
                        width=4
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Data Quality Visualizations",
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Missing Values",
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(id='missing-values-plot'),
                                                type="circle"
                                            )
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Data Distribution",
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(id='distribution-plot'),
                                                type="circle"
                                            )
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Correlation Matrix",
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(id='correlation-plot'),
                                                type="circle"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        icon="chart-area"
                    ),
                    width=12
                )
            )
        ],
        fluid=True,
        style={'paddingTop': '100px', 'maxWidth': '1400px'}
    )

def create_clustering_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Clustering Analysis",
                            style={
                                **CUSTOM_STYLES['header'],
                                'fontSize': '32px',
                                'marginBottom': '12px'
                            }
                        ),
                        html.P(
                            "Discover patterns and natural groupings in your data",
                            style={
                                **CUSTOM_STYLES['text'],
                                'fontSize': '16px',
                                'color': '#666'
                            }
                        )
                    ],
                    width=12
                ),
                className="mb-5"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Clustering Configuration",
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Clustering Algorithm", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                                dcc.Dropdown(
                                                    id='clustering-algorithm',
                                                    options=[
                                                        {'label': 'K-means clustering', 'value': 'kmeans'},
                                                        {'label': 'DBSCAN clustering', 'value': 'dbscan'},
                                                        {'label': 'Hierarchical clustering', 'value': 'hierarchical'}
                                                    ],
                                                    value='kmeans',
                                                    style={'marginBottom': '20px'}
                                                )
                                            ],
                                            width=6
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Number of Clusters", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                                dcc.Slider(
                                                    id='n-clusters',
                                                    min=2,
                                                    max=15,
                                                    step=1,
                                                    value=4,
                                                    marks={i: str(i) for i in range(2, 16, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": False}
                                                )
                                            ],
                                            width=6
                                        )
                                    ],
                                    className="mb-4"
                                ),
                                
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Dimensionality Reduction", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                                dcc.Dropdown(
                                                    id='dimred-method-clustering',
                                                    options=[
                                                        {'label': 'Principal Component Analysis', 'value': 'PCA'},
                                                        {'label': 'Uniform Manifold Approximation', 'value': 'UMAP'}
                                                    ],
                                                    value='PCA',
                                                    style={'marginBottom': '20px'}
                                                )
                                            ],
                                            width=6
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                                dbc.Button(
                                                    [
                                                        html.I(className="fas fa-project-diagram me-2"),
                                                        "Execute Clustering Analysis"
                                                    ],
                                                    id="run-clustering",
                                                    color="primary",
                                                    className="w-100 mt-2",
                                                    style=CUSTOM_STYLES['primary_button']
                                                )
                                            ],
                                            width=6
                                        )
                                    ]
                                )
                            ],
                            icon="project-diagram"
                        ),
                        width=12
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Clustering Visualization",
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            label="2D Projection",
                                            children=[
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id='clustering-2d-plot',
                                                        style={'height': PlatformConfig.PLOT_HEIGHT}
                                                    ),
                                                    type="circle"
                                                )
                                            ]
                                        ),
                                        dbc.Tab(
                                            label="3D Projection",
                                            children=[
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id='clustering-3d-plot',
                                                        style={'height': PlatformConfig.PLOT_HEIGHT}
                                                    ),
                                                    type="circle"
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ],
                            icon="eye"
                        ),
                        width=8
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Clustering Metrics",
                            [
                                html.Div(id='clustering-metrics'),
                                dcc.Loading(
                                    dcc.Graph(id='clustering-metrics-plot'),
                                    type="circle"
                                ),
                                html.Hr(style={'margin': '20px 0'}),
                                html.Div(id='cluster-statistics')
                            ],
                            icon="chart-line"
                        ),
                        width=4
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Cluster Analysis Details",
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Cluster Statistics",
                                        children=[
                                            dash_table.DataTable(
                                                id='cluster-stats-table',
                                                page_size=10,
                                                style_table={'overflowX': 'auto'},
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'padding': '12px',
                                                    'fontFamily': "'Inter', sans-serif"
                                                },
                                                style_header={
                                                    'backgroundColor': '#f8f9fa',
                                                    'fontWeight': '600',
                                                    'borderBottom': '2px solid #e0e0e0'
                                                },
                                                style_data_conditional=[
                                                    {
                                                        'if': {'row_index': 'odd'},
                                                        'backgroundColor': '#fafafa'
                                                    }
                                                ]
                                            )
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Silhouette Analysis",
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(id='silhouette-plot'),
                                                type="circle"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        icon="table"
                    ),
                    width=12
                )
            )
        ],
        fluid=True,
        style={'paddingTop': '100px', 'maxWidth': '1400px'}
    )

def create_anomaly_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Anomaly Detection",
                            style={
                                **CUSTOM_STYLES['header'],
                                'fontSize': '32px',
                                'marginBottom': '12px'
                            }
                        ),
                        html.P(
                            "Identify outliers and anomalous patterns in your data",
                            style={
                                **CUSTOM_STYLES['text'],
                                'fontSize': '16px',
                                'color': '#666'
                            }
                        )
                    ],
                    width=12
                ),
                className="mb-5"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Anomaly Detection Configuration",
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Detection Method", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                                dcc.Dropdown(
                                                    id='anomaly-method',
                                                    options=[
                                                        {'label': 'Isolation Forest', 'value': 'isolation_forest'},
                                                        {'label': 'Local Outlier Factor', 'value': 'lof'},
                                                        {'label': 'One-Class SVM', 'value': 'svm'}
                                                    ],
                                                    value='isolation_forest',
                                                    style={'marginBottom': '20px'}
                                                )
                                            ],
                                            width=6
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Contamination Rate", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                                dcc.Slider(
                                                    id='contamination-rate',
                                                    min=0.01,
                                                    max=0.5,
                                                    step=0.01,
                                                    value=0.1,
                                                    marks={0.01: '1%', 0.1: '10%', 0.2: '20%', 0.3: '30%', 0.4: '40%', 0.5: '50%'},
                                                    tooltip={"placement": "bottom", "always_visible": False}
                                                ),
                                                html.Small(
                                                    "Expected proportion of anomalies in the data",
                                                    style={'color': '#999', 'fontSize': '12px', 'display': 'block', 'marginTop': '8px'}
                                                )
                                            ],
                                            width=6
                                        )
                                    ],
                                    className="mb-4"
                                ),
                                
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-search me-2"),
                                        "Detect Anomalies"
                                    ],
                                    id="detect-anomalies",
                                    color="primary",
                                    className="w-100",
                                    style=CUSTOM_STYLES['primary_button']
                                )
                            ],
                            icon="cogs"
                        ),
                        width=12
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Anomaly Distribution",
                            [
                                dcc.Loading(
                                    dcc.Graph(
                                        id='anomaly-distribution-plot',
                                        style={'height': PlatformConfig.PLOT_HEIGHT}
                                    ),
                                    type="circle"
                                )
                            ],
                            icon="chart-bar"
                        ),
                        width=6
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Anomaly Detection Results",
                            [
                                html.Div(id='anomaly-summary'),
                                dcc.Loading(
                                    dash_table.DataTable(
                                        id='anomaly-table',
                                        page_size=10,
                                        style_table={'overflowX': 'auto', 'height': '400px'},
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '10px',
                                            'fontSize': '13px',
                                            'fontFamily': "'Inter', sans-serif"
                                        },
                                        style_header={
                                            'backgroundColor': '#f8f9fa',
                                            'fontWeight': '600',
                                            'borderBottom': '2px solid #e0e0e0'
                                        }
                                    ),
                                    type="circle"
                                )
                            ],
                            icon="table"
                        ),
                        width=6
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Anomaly Visualization",
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Anomaly Scores",
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id='anomaly-score-plot',
                                                    style={'height': PlatformConfig.PLOT_HEIGHT}
                                                ),
                                                type="circle"
                                            )
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Anomaly Heatmap",
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id='anomaly-heatmap',
                                                    style={'height': PlatformConfig.PLOT_HEIGHT}
                                                ),
                                                type="circle"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        icon="fire"
                    ),
                    width=12
                )
            )
        ],
        fluid=True,
        style={'paddingTop': '100px', 'maxWidth': '1400px'}
    )

def create_insights_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "AI-Driven Insights",
                            style={
                                **CUSTOM_STYLES['header'],
                                'fontSize': '32px',
                                'marginBottom': '12px'
                            }
                        ),
                        html.P(
                            "Comprehensive analytical insights generated from your data",
                            style={
                                **CUSTOM_STYLES['text'],
                                'fontSize': '16px',
                                'color': '#666'
                            }
                        )
                    ],
                    width=12
                ),
                className="mb-5"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Executive Summary",
                        [
                            html.Div(id='executive-summary'),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-sync-alt me-2"),
                                    "Generate Insights"
                                ],
                                id="generate-insights",
                                color="primary",
                                className="mt-3",
                                style=CUSTOM_STYLES['primary_button']
                            )
                        ],
                        icon="file-alt"
                    ),
                    width=12
                ),
                className="mb-4"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Data Quality Insights",
                            [
                                html.Div(id='data-quality-insights')
                            ],
                            icon="check-circle",
                            color="success"
                        ),
                        width=6
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Clustering Insights",
                            [
                                html.Div(id='clustering-insights')
                            ],
                            icon="project-diagram",
                            color="info"
                        ),
                        width=6
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Anomaly Insights",
                            [
                                html.Div(id='anomaly-insights')
                            ],
                            icon="exclamation-triangle",
                            color="warning"
                        ),
                        width=6
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Recommendations",
                            [
                                html.Div(id='recommendations-insights')
                            ],
                            icon="lightbulb",
                            color="primary"
                        ),
                        width=6
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Detailed Analysis Report",
                        [
                            dash_table.DataTable(
                                id='insights-table',
                                columns=[
                                    {'name': 'Category', 'id': 'category'},
                                    {'name': 'Title', 'id': 'title'},
                                    {'name': 'Content', 'id': 'content'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '12px',
                                    'fontSize': '13px',
                                    'fontFamily': "'Inter', sans-serif",
                                    'whiteSpace': 'normal',
                                    'height': 'auto'
                                },
                                style_header={
                                    'backgroundColor': '#f8f9fa',
                                    'fontWeight': '600',
                                    'borderBottom': '2px solid #e0e0e0'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': '#fafafa'
                                    }
                                ]
                            )
                        ],
                        icon="clipboard-list"
                    ),
                    width=12
                )
            )
        ],
        fluid=True,
        style={'paddingTop': '100px', 'maxWidth': '1400px'}
    )

def create_export_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Export & Session Management",
                            style={
                                **CUSTOM_STYLES['header'],
                                'fontSize': '32px',
                                'marginBottom': '12px'
                            }
                        ),
                        html.P(
                            "Export analysis results and manage your session",
                            style={
                                **CUSTOM_STYLES['text'],
                                'fontSize': '16px',
                                'color': '#666'
                            }
                        )
                    ],
                    width=12
                ),
                className="mb-5"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        create_card(
                            "Export Configuration",
                            [
                                html.Label("Select Data to Export", style={'fontWeight': '600', 'marginBottom': '12px'}),
                                dbc.Checklist(
                                    id='export-options',
                                    options=[
                                        {'label': ' Processed dataset', 'value': 'processed_data'},
                                        {'label': ' Clustering results', 'value': 'clustering_results'},
                                        {'label': ' Anomaly scores', 'value': 'anomaly_scores'},
                                        {'label': ' PCA embeddings', 'value': 'pca_embeddings'},
                                        {'label': ' Analysis insights', 'value': 'analysis_insights'},
                                        {'label': ' Pipeline log', 'value': 'pipeline_log'}
                                    ],
                                    value=['processed_data', 'clustering_results'],
                                    inline=False,
                                    className="mb-4"
                                ),
                                
                                html.Label("Export Format", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                dcc.Dropdown(
                                    id='export-format',
                                    options=[
                                        {'label': 'Excel workbook (.xlsx)', 'value': 'excel'},
                                        {'label': 'CSV files (.zip)', 'value': 'csv'},
                                        {'label': 'JSON format (.json)', 'value': 'json'}
                                    ],
                                    value='excel',
                                    style={'marginBottom': '20px'}
                                ),
                                
                                html.Label("File Name", style={'fontWeight': '600', 'marginBottom': '8px'}),
                                dbc.Input(
                                    id='export-filename',
                                    type='text',
                                    value=f"bioinformatics_analysis_{data_manager.session_id}",
                                    style={'marginBottom': '30px'}
                                ),
                                
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download me-2"),
                                        "Export Selected Data"
                                    ],
                                    id="export-data",
                                    color="primary",
                                    className="w-100",
                                    style=CUSTOM_STYLES['primary_button']
                                )
                            ],
                            icon="download"
                        ),
                        width=6
                    ),
                    
                    dbc.Col(
                        create_card(
                            "Session Management",
                            [
                                dbc.Alert(
                                    [
                                        html.Div(
                                            [
                                                html.I(className="fas fa-info-circle me-2"),
                                                html.Strong("Current Session Information")
                                            ],
                                            className="d-flex align-items-center mb-2"
                                        ),
                                        html.P(f"Session ID: {data_manager.session_id}"),
                                        html.P(f"Data loaded: {'Yes' if data_manager.raw_data is not None else 'No'}"),
                                        html.P(f"Analysis steps completed: {len(data_manager.pipeline_log)}"),
                                        html.Hr(),
                                        html.P(
                                            "Warning: Clearing session will permanently remove all uploaded data and analysis results.",
                                            className="mb-0 text-danger"
                                        )
                                    ],
                                    color="info",
                                    className="mb-4"
                                ),
                                
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="fas fa-trash-alt me-2"),
                                                    "Clear Session"
                                                ],
                                                id="clear-session",
                                                color="danger",
                                                className="w-100",
                                                style={**CUSTOM_STYLES['primary_button'], 'backgroundColor': '#dc3545'}
                                            ),
                                            width=6
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="fas fa-file-download me-2"),
                                                    "Download Pipeline Log"
                                                ],
                                                id="download-log",
                                                color="secondary",
                                                className="w-100",
                                                style=CUSTOM_STYLES['secondary_button']
                                            ),
                                            width=6
                                        )
                                    ]
                                )
                            ],
                            icon="cog"
                        ),
                        width=6
                    )
                ],
                className="mb-4"
            ),
            
            dbc.Row(
                dbc.Col(
                    create_card(
                        "Pipeline Execution Log",
                        [
                            dash_table.DataTable(
                                id='pipeline-log-table',
                                columns=[
                                    {'name': 'Timestamp', 'id': 'timestamp'},
                                    {'name': 'Step', 'id': 'step'},
                                    {'name': 'Status', 'id': 'status'},
                                    {'name': 'Message', 'id': 'message'},
                                    {'name': 'Time (s)', 'id': 'execution_time'}
                                ],
                                data=data_manager.pipeline_log[-50:],
                                page_size=10,
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px',
                                    'fontSize': '12px',
                                    'fontFamily': "'Inter', sans-serif"
                                },
                                style_header={
                                    'backgroundColor': '#f8f9fa',
                                    'fontWeight': '600',
                                    'borderBottom': '2px solid #e0e0e0'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{status} = "completed"'},
                                        'backgroundColor': '#d4edda',
                                        'color': '#155724'
                                    },
                                    {
                                        'if': {'filter_query': '{status} = "failed"'},
                                        'backgroundColor': '#f8d7da',
                                        'color': '#721c24'
                                    },
                                    {
                                        'if': {'filter_query': '{status} = "started"'},
                                        'backgroundColor': '#d1ecf1',
                                        'color': '#0c5460'
                                    }
                                ]
                            )
                        ],
                        icon="history"
                    ),
                    width=12
                )
            )
        ],
        fluid=True,
        style={'paddingTop': '100px', 'maxWidth': '1400px'}
    )

# ============================================
# APP LAYOUT
# ============================================

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-store'),
        dcc.Download(id="download-data"),
        
        create_navigation(),
        
        html.Div(
            id='page-content',
            style={'paddingTop': '100px', 'minHeight': 'calc(100vh - 200px)'}
        ),
        
        create_footer(),
        
        html.Div(id='hidden-div', style={'display': 'none'})
    ],
    style=CUSTOM_STYLES['background']
)

# ============================================
# CALLBACKS
# ============================================

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/' or pathname == '/upload':
        return create_upload_page()
    elif pathname == '/quality':
        return create_quality_page()
    elif pathname == '/clustering':
        return create_clustering_page()
    elif pathname == '/anomaly':
        return create_anomaly_page()
    elif pathname == '/insights':
        return create_insights_page()
    elif pathname == '/export':
        return create_export_page()
    else:
        return create_upload_page()

@app.callback(
    [Output('upload-status', 'children'),
     Output('session-store', 'data')],
    [Input('upload-data', 'contents'),
     Input('load-sample-data', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def handle_file_upload(contents, sample_clicks, filename, last_modified):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == 'load-sample-data' and sample_clicks:
        try:
            # Create comprehensive sample dataset
            np.random.seed(42)
            n_samples = 500
            n_genes = 2000
            
            # Create synthetic expression data with 4 clusters
            data = []
            cluster_labels = []
            
            for i in range(4):
                cluster_data = np.random.normal(
                    loc=i * 3,
                    scale=1.0 + i * 0.2,
                    size=(n_samples // 4, n_genes)
                )
                
                # Add cluster-specific marker genes
                marker_indices = np.random.choice(n_genes, 100, replace=False)
                cluster_data[:, marker_indices] += np.random.normal(5, 1, (n_samples // 4, 100))
                
                data.append(cluster_data)
                cluster_labels.extend([f'Cluster_{i}'] * (n_samples // 4))
            
            sample_data = np.vstack(data)
            
            # Create gene names
            genes = [f'Gene_{i:05d}' for i in range(n_genes)]
            samples = [f'Sample_{i:04d}' for i in range(n_samples)]
            
            df = pd.DataFrame(
                sample_data,
                index=samples,
                columns=genes
            )
            
            # Add metadata
            df['Cluster'] = cluster_labels
            df['Batch'] = np.random.choice(['Batch_A', 'Batch_B', 'Batch_C'], n_samples)
            df['Condition'] = np.random.choice(['Healthy', 'Disease'], n_samples, p=[0.7, 0.3])
            df['Survival_Time'] = np.random.exponential(365 * 2, n_samples)
            df['Event'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            
            data_manager.raw_data = df.reset_index().rename(columns={'index': 'Sample_ID'})
            
            data_manager.log_pipeline_step(
                'sample_data_loaded',
                'completed',
                f'Loaded sample dataset with {n_samples} samples and {n_genes} genes'
            )
            
            data_manager.add_data_insight(
                'sample_data',
                f"Loaded sample bioinformatics dataset: {n_samples} samples, {n_genes} genes",
                'info'
            )
            
            alert = dbc.Alert(
                [
                    html.Div(
                        [
                            html.I(className="fas fa-check-circle me-2", style={'color': '#28a745'}),
                            html.Strong("Sample Data Loaded Successfully", className="alert-heading")
                        ],
                        className="d-flex align-items-center mb-2"
                    ),
                    html.P(f"Samples: {n_samples:,}, Genes: {n_genes:,}"),
                    html.P("A synthetic bioinformatics dataset has been loaded for analysis."),
                    dbc.Button(
                        "Proceed to Data Quality Analysis",
                        href="/quality",
                        color="success",
                        className="mt-2"
                    )
                ],
                color="success",
                dismissable=True,
                style={'borderLeft': '4px solid #28a745'}
            )
            
            return alert, {'sample_loaded': True}
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return create_data_quality_alert(
                'error',
                'Sample Data Error',
                f"Failed to load sample data: {str(e)}"
            ), dash.no_update
    
    elif triggered_id == 'upload-data' and contents:
        try:
            success, result = data_manager.universal_data_loader(contents, filename)
            
            if success:
                # Create success alert with data summary
                alert_content = [
                    html.Div(
                        [
                            html.I(className="fas fa-check-circle me-2", style={'color': '#28a745'}),
                            html.Strong("File Uploaded Successfully", className="alert-heading")
                        ],
                        className="d-flex align-items-center mb-2"
                    ),
                    html.P(f"File: {filename}"),
                    html.P(f"Rows: {result['rows']:,}, Columns: {result['columns']:,}"),
                    html.P(f"Numeric features: {result['numeric_columns']}, Categorical: {result['categorical_columns']}"),
                    
                    dbc.Button(
                        "Proceed to Data Quality Analysis",
                        href="/quality",
                        color="success",
                        className="mt-2"
                    )
                ]
                
                # Add warnings if needed
                if result['missing_values'] > 0:
                    alert_content.insert(3, html.P(
                        f"Missing values detected: {result['missing_values']:,} ({result['missing_percentage']:.1f}%)",
                        style={'color': '#ffc107', 'fontWeight': '500'}
                    ))
                
                if result['duplicate_rows'] > 0:
                    alert_content.insert(4, html.P(
                        f"Duplicate rows: {result['duplicate_rows']:,}",
                        style={'color': '#ffc107', 'fontWeight': '500'}
                    ))
                
                alert = dbc.Alert(
                    alert_content,
                    color="success",
                    dismissable=True,
                    style={'borderLeft': '4px solid #28a745'}
                )
                
                return alert, {'file_uploaded': True}
            else:
                return create_data_quality_alert(
                    'error',
                    'Upload Failed',
                    f"Could not process the file: {result}"
                ), dash.no_update
                
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return create_data_quality_alert(
                'error',
                'Upload Error',
                f"Failed to process file: {str(e)}"
            ), dash.no_update
    
    return "", dash.no_update

@app.callback(
    [Output('quality-metrics', 'children'),
     Output('data-insights-container', 'children'),
     Output('missing-values-plot', 'figure'),
     Output('distribution-plot', 'figure'),
     Output('correlation-plot', 'figure')],
    [Input('run-preprocessing', 'n_clicks')],
    [State('imputation-strategy', 'value'),
     State('normalization-method', 'value'),
     State('outlier-method', 'value')]
)
def run_preprocessing_and_quality(n_clicks, imputation_strategy, normalization_method, outlier_method):
    if n_clicks is None or data_manager.raw_data is None:
        return "", "", go.Figure(), go.Figure(), go.Figure()
    
    try:
        # Run preprocessing
        success, summary = data_manager.adaptive_preprocessing_pipeline()
        
        if success:
            # Create quality metrics cards
            metrics_row = dbc.Row([
                dbc.Col(
                    create_metric_card(
                        "Samples",
                        f"{summary['rows_processed']:,}",
                        icon="users"
                    ),
                    width=3
                ),
                dbc.Col(
                    create_metric_card(
                        "Features",
                        f"{summary['columns_processed']:,}",
                        icon="layer-group"
                    ),
                    width=3
                ),
                dbc.Col(
                    create_metric_card(
                        "Missing Values",
                        f"{summary['missing_values_handled']:,}",
                        icon="exclamation-triangle"
                    ),
                    width=3
                ),
                dbc.Col(
                    create_metric_card(
                        "Duplicates",
                        f"{summary['duplicates_removed']:,}",
                        icon="copy"
                    ),
                    width=3
                )
            ])
            
            # Create data insights
            insights_content = []
            for insight in data_manager.data_insights[-5:]:  # Show last 5 insights
                severity_color = {
                    'error': 'danger',
                    'warning': 'warning',
                    'info': 'info',
                    'success': 'success'
                }.get(insight['severity'], 'info')
                
                insights_content.append(
                    create_data_quality_alert(
                        severity_color,
                        insight['type'].replace('_', ' ').title(),
                        insight['message']
                    )
                )
            
            # Create visualizations
            if data_manager.raw_data is not None:
                # Missing values heatmap
                missing_fig = px.imshow(
                    data_manager.raw_data.isnull().astype(int).T,
                    title="Missing Values Heatmap",
                    color_continuous_scale='Reds',
                    labels={'color': 'Missing'}
                )
                missing_fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Inter'}
                )
                
                # Distribution plot
                numeric_cols = data_manager.raw_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    dist_fig = px.histogram(
                        data_manager.raw_data[numeric_cols[0]].dropna(),
                        title=f"Distribution of {numeric_cols[0]}",
                        marginal="box",
                        nbins=50
                    )
                    dist_fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font={'family': 'Inter'}
                    )
                else:
                    dist_fig = go.Figure()
                    dist_fig.add_annotation(
                        text="No numeric columns available for distribution plot",
                        showarrow=False,
                        font={'size': 14}
                    )
                
                # Correlation plot
                if len(numeric_cols) > 1:
                    # Calculate correlation for first 10 numeric columns
                    corr_data = data_manager.raw_data[numeric_cols[:10]].corr()
                    corr_fig = px.imshow(
                        corr_data,
                        title="Feature Correlation Matrix",
                        color_continuous_scale='RdBu',
                        zmin=-1, zmax=1,
                        labels={'color': 'Correlation'}
                    )
                    corr_fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font={'family': 'Inter'}
                    )
                else:
                    corr_fig = go.Figure()
                    corr_fig.add_annotation(
                        text="Not enough numeric features for correlation matrix",
                        showarrow=False,
                        font={'size': 14}
                    )
                
                return metrics_row, insights_content, missing_fig, dist_fig, corr_fig
            
        else:
            return create_data_quality_alert(
                'error',
                'Preprocessing Failed',
                summary
            ), "", go.Figure(), go.Figure(), go.Figure()
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        error_alert = create_data_quality_alert(
            'error',
            'Processing Error',
            f"An error occurred during preprocessing: {str(e)}"
        )
        return error_alert, "", go.Figure(), go.Figure(), go.Figure()
    
    return "", "", go.Figure(), go.Figure(), go.Figure()

@app.callback(
    [Output('clustering-2d-plot', 'figure'),
     Output('clustering-3d-plot', 'figure'),
     Output('clustering-metrics', 'children'),
     Output('clustering-metrics-plot', 'figure'),
     Output('cluster-statistics', 'children'),
     Output('cluster-stats-table', 'data'),
     Output('silhouette-plot', 'figure')],
    [Input('run-clustering', 'n_clicks')],
    [State('clustering-algorithm', 'value'),
     State('n-clusters', 'value'),
     State('dimred-method-clustering', 'value')]
)
def run_clustering_analysis(n_clicks, algorithm, n_clusters, dimred_method):
    if n_clicks is None or data_manager.processed_data is None:
        return go.Figure(), go.Figure(), "", go.Figure(), "", [], go.Figure()
    
    try:
        # First perform dimensionality reduction
        success_dimred, _ = data_manager.perform_dimensionality_reduction(dimred_method, 3)
        if not success_dimred:
            raise ValueError("Dimensionality reduction failed")
        
        # Perform clustering
        success_cluster, result = data_manager.perform_clustering(algorithm, n_clusters)
        
        if success_cluster:
            cluster_result = data_manager.clustering_results.get(algorithm)
            if cluster_result is None:
                raise ValueError("Clustering results not found")
            
            labels = cluster_result['labels']
            embeddings = data_manager.embeddings.get(dimred_method)
            
            if embeddings is None:
                raise ValueError(f"{dimred_method} embeddings not found")
            
            # Create 2D plot
            if embeddings.shape[1] >= 2:
                plot_2d_df = embeddings.iloc[:, :2].copy()
                plot_2d_df['Cluster'] = labels.astype(str)
                
                fig_2d = px.scatter(
                    plot_2d_df,
                    x=plot_2d_df.columns[0],
                    y=plot_2d_df.columns[1],
                    color='Cluster',
                    title=f"{algorithm.upper()} Clustering - 2D {dimred_method} Projection",
                    color_discrete_sequence=PlatformConfig.get_color_palette(len(np.unique(labels))),
                    labels={'color': 'Cluster'}
                )
                fig_2d.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Inter'},
                    height=PlatformConfig.PLOT_HEIGHT
                )
            else:
                fig_2d = go.Figure()
                fig_2d.add_annotation(
                    text="Not enough dimensions for 2D plot",
                    showarrow=False,
                    font={'size': 14}
                )
            
            # Create 3D plot
            if embeddings.shape[1] >= 3:
                plot_3d_df = embeddings.iloc[:, :3].copy()
                plot_3d_df['Cluster'] = labels.astype(str)
                
                fig_3d = px.scatter_3d(
                    plot_3d_df,
                    x=plot_3d_df.columns[0],
                    y=plot_3d_df.columns[1],
                    z=plot_3d_df.columns[2],
                    color='Cluster',
                    title=f"{algorithm.upper()} Clustering - 3D {dimred_method} Projection",
                    color_discrete_sequence=PlatformConfig.get_color_palette(len(np.unique(labels)))
                )
                fig_3d.update_layout(
                    scene=dict(bgcolor='white'),
                    paper_bgcolor='white',
                    font={'family': 'Inter'},
                    height=PlatformConfig.PLOT_HEIGHT
                )
            else:
                fig_3d = go.Figure()
                fig_3d.add_annotation(
                    text="Not enough dimensions for 3D plot",
                    showarrow=False,
                    font={'size': 14}
                )
            
            # Create metrics display
            metrics_content = [
                html.H6("Clustering Performance", style={'fontWeight': '600', 'marginBottom': '10px'}),
                html.P(f"Algorithm: {algorithm.upper()}", style={'marginBottom': '5px'}),
                html.P(f"Clusters detected: {result['n_clusters']}", style={'marginBottom': '5px'}),
                html.P(f"Silhouette score: {result['silhouette_score']:.3f}", style={'marginBottom': '5px'}),
                html.P(f"Samples clustered: {result['labels_assigned']:,}", style={'marginBottom': '5px'})
            ]
            
            # Create metrics plot
            if 'cluster_stats' in cluster_result:
                sizes = [stats['size'] for stats in cluster_result['cluster_stats'].values()]
                clusters = [f"Cluster {label}" for label in cluster_result['cluster_stats'].keys()]
                
                metrics_fig = px.bar(
                    x=clusters,
                    y=sizes,
                    title="Cluster Sizes",
                    color=clusters,
                    color_discrete_sequence=PlatformConfig.get_color_palette(len(clusters))
                )
                metrics_fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Inter'},
                    showlegend=False,
                    height=250
                )
            else:
                metrics_fig = go.Figure()
                metrics_fig.add_annotation(
                    text="Cluster statistics not available",
                    showarrow=False,
                    font={'size': 14}
                )
            
            # Create cluster statistics
            if 'cluster_stats' in cluster_result:
                stats_content = []
                for label, stats in cluster_result['cluster_stats'].items():
                    stats_content.append(
                        html.Div(
                            [
                                html.Strong(f"Cluster {label}: ", style={'color': PlatformConfig.PRIMARY_COLOR}),
                                html.Span(f"{stats['size']} samples ({stats['percentage']:.1f}%)")
                            ],
                            className="mb-2"
                        )
                    )
            else:
                stats_content = [html.P("Cluster statistics not available")]
            
            # Create cluster statistics table
            table_data = []
            if 'cluster_stats' in cluster_result:
                for label, stats in cluster_result['cluster_stats'].items():
                    table_data.append({
                        'Cluster': f"Cluster {label}",
                        'Size': stats['size'],
                        'Percentage': f"{stats['percentage']:.1f}%",
                        'Spread': f"{stats['spread']:.3f}"
                    })
            
            # Create silhouette plot
            if result['n_clusters'] > 1:
                silhouette_vals = silhouette_samples(embeddings.values[:, :2], labels)
                silhouette_df = pd.DataFrame({
                    'Cluster': labels,
                    'Silhouette': silhouette_vals
                })
                
                silhouette_fig = px.box(
                    silhouette_df,
                    x='Cluster',
                    y='Silhouette',
                    title="Silhouette Scores by Cluster",
                    color='Cluster',
                    color_discrete_sequence=PlatformConfig.get_color_palette(len(np.unique(labels)))
                )
                silhouette_fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Inter'},
                    showlegend=False,
                    height=400
                )
            else:
                silhouette_fig = go.Figure()
                silhouette_fig.add_annotation(
                    text="Silhouette analysis requires at least 2 clusters",
                    showarrow=False,
                    font={'size': 14}
                )
            
            return fig_2d, fig_3d, metrics_content, metrics_fig, stats_content, table_data, silhouette_fig
        
        else:
            error_msg = f"Clustering failed: {result}"
            error_fig = go.Figure()
            error_fig.add_annotation(text=error_msg, showarrow=False, font={'size': 14})
            return error_fig, error_fig, create_data_quality_alert('error', 'Clustering Error', result), go.Figure(), "", [], go.Figure()
            
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(text=f"Clustering error: {str(e)}", showarrow=False, font={'size': 14})
        return error_fig, error_fig, create_data_quality_alert('error', 'Clustering Error', str(e)), go.Figure(), "", [], go.Figure()
    
    return go.Figure(), go.Figure(), "", go.Figure(), "", [], go.Figure()

@app.callback(
    [Output('anomaly-distribution-plot', 'figure'),
     Output('anomaly-summary', 'children'),
     Output('anomaly-table', 'data'),
     Output('anomaly-score-plot', 'figure'),
     Output('anomaly-heatmap', 'figure')],
    [Input('detect-anomalies', 'n_clicks')],
    [State('anomaly-method', 'value'),
     State('contamination-rate', 'value')]
)
def run_anomaly_detection(n_clicks, anomaly_method, contamination_rate):
    if n_clicks is None or data_manager.processed_data is None:
        return go.Figure(), "", [], go.Figure(), go.Figure()
    
    try:
        success, summary = data_manager.detect_anomalies(contamination_rate)
        
        if success and data_manager.anomaly_scores is not None:
            anomaly_scores = data_manager.anomaly_scores
            
            # Create distribution plot
            dist_fig = px.histogram(
                anomaly_scores,
                x='anomaly_score',
                color='is_anomaly',
                title="Anomaly Score Distribution",
                color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
                nbins=50,
                labels={'is_anomaly': 'Is Anomaly'}
            )
            dist_fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'family': 'Inter'},
                height=PlatformConfig.PLOT_HEIGHT
            )
            
            # Create summary
            n_anomalies = int(anomaly_scores['is_anomaly'].sum())
            n_total = len(anomaly_scores)
            anomaly_pct = (n_anomalies / n_total) * 100
            
            summary_content = [
                html.H6("Anomaly Detection Summary", style={'fontWeight': '600', 'marginBottom': '10px'}),
                html.P(f"Total samples: {n_total:,}", style={'marginBottom': '5px'}),
                html.P(f"Anomalies detected: {n_anomalies:,}", style={'marginBottom': '5px'}),
                html.P(f"Anomaly percentage: {anomaly_pct:.1f}%", style={'marginBottom': '5px'}),
                html.P(f"Detection method: Isolation Forest", style={'marginBottom': '5px'})
            ]
            
            # Create anomaly table
            if data_manager.processed_data is not None:
                anomaly_df = data_manager.processed_data.copy()
                anomaly_df['Anomaly_Score'] = anomaly_scores['anomaly_score']
                anomaly_df['Is_Anomaly'] = anomaly_scores['is_anomaly']
                anomaly_df['Percentile'] = anomaly_scores['anomaly_percentile']
                
                top_anomalies = anomaly_df.nlargest(10, 'Anomaly_Score')
                table_data = top_anomalies.head(10).to_dict('records')
            else:
                table_data = []
            
            # Create anomaly score plot
            score_fig = px.scatter(
                anomaly_scores.reset_index(),
                x=anomaly_scores.reset_index().index,
                y='anomaly_score',
                color='is_anomaly',
                title="Anomaly Scores by Sample Index",
                color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
                labels={'index': 'Sample Index', 'is_anomaly': 'Is Anomaly'}
            )
            score_fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'family': 'Inter'},
                height=PlatformConfig.PLOT_HEIGHT
            )
            
            # Create anomaly heatmap
            if data_manager.processed_data is not None:
                numeric_cols = data_manager.processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    anomaly_indices = anomaly_scores[anomaly_scores['is_anomaly']].index
                    if len(anomaly_indices) > 0:
                        anomaly_data = data_manager.processed_data.loc[anomaly_indices, numeric_cols[:10]].T
                        heatmap_fig = px.imshow(
                            anomaly_data,
                            title="Anomaly Feature Heatmap",
                            color_continuous_scale='RdBu',
                            labels={'color': 'Feature Value'}
                        )
                        heatmap_fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font={'family': 'Inter'},
                            height=PlatformConfig.PLOT_HEIGHT
                        )
                    else:
                        heatmap_fig = go.Figure()
                        heatmap_fig.add_annotation(
                            text="No anomalies detected for heatmap",
                            showarrow=False,
                            font={'size': 14}
                        )
                else:
                    heatmap_fig = go.Figure()
                    heatmap_fig.add_annotation(
                        text="No numeric features available for heatmap",
                        showarrow=False,
                        font={'size': 14}
                    )
            else:
                heatmap_fig = go.Figure()
                heatmap_fig.add_annotation(
                    text="No processed data available",
                    showarrow=False,
                    font={'size': 14}
                )
            
            return dist_fig, summary_content, table_data, score_fig, heatmap_fig
        
        else:
            error_fig = go.Figure()
            error_fig.add_annotation(text="Anomaly detection failed", showarrow=False, font={'size': 14})
            return error_fig, create_data_quality_alert('error', 'Anomaly Detection Failed', summary), [], go.Figure(), go.Figure()
            
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(text=f"Anomaly detection error: {str(e)}", showarrow=False, font={'size': 14})
        return error_fig, create_data_quality_alert('error', 'Anomaly Detection Error', str(e)), [], go.Figure(), go.Figure()
    
    return go.Figure(), "", [], go.Figure(), go.Figure()

@app.callback(
    [Output('executive-summary', 'children'),
     Output('data-quality-insights', 'children'),
     Output('clustering-insights', 'children'),
     Output('anomaly-insights', 'children'),
     Output('recommendations-insights', 'children'),
     Output('insights-table', 'data')],
    [Input('generate-insights', 'n_clicks')]
)
def generate_insights(n_clicks):
    if n_clicks is None:
        return "", "", "", "", "", []
    
    try:
        insights = data_manager.generate_comprehensive_insights()
        
        # Group insights by category
        executive_insights = []
        data_quality_insights = []
        clustering_insights = []
        anomaly_insights = []
        recommendation_insights = []
        
        for insight in insights:
            if insight['category'] == 'Data Quality':
                data_quality_insights.append(insight)
            elif insight['category'] == 'Clustering Analysis':
                clustering_insights.append(insight)
            elif insight['category'] == 'Anomaly Detection':
                anomaly_insights.append(insight)
            elif insight['category'] == 'Recommendations':
                recommendation_insights.append(insight)
            else:
                executive_insights.append(insight)
        
        # Create executive summary
        exec_summary = []
        if data_manager.data_summary:
            exec_summary.append(
                html.P(
                    f"The analysis processed {data_manager.data_summary.get('rows', 0):,} samples with "
                    f"{data_manager.data_summary.get('columns', 0):,} features.",
                    style={'marginBottom': '10px'}
                )
            )
        
        if data_manager.clustering_results:
            exec_summary.append(
                html.P(
                    f"Clustering analysis identified {len(data_manager.clustering_results)} pattern(s) in the data.",
                    style={'marginBottom': '10px'}
                )
            )
        
        if data_manager.anomaly_scores is not None:
            anomaly_count = data_manager.anomaly_scores['is_anomaly'].sum()
            exec_summary.append(
                html.P(
                    f"Anomaly detection identified {anomaly_count:,} potentially anomalous samples.",
                    style={'marginBottom': '10px'}
                )
            )
        
        # Create data quality insights display
        data_quality_content = []
        for insight in data_quality_insights[:3]:
            data_quality_content.append(
                html.Div(
                    [
                        html.Strong(insight['title'], style={'display': 'block', 'marginBottom': '5px'}),
                        html.P(insight['content'], style={'fontSize': '14px', 'marginBottom': '10px'})
                    ],
                    className="mb-3"
                )
            )
        
        # Create clustering insights display
        clustering_content = []
        for insight in clustering_insights[:3]:
            clustering_content.append(
                html.Div(
                    [
                        html.Strong(insight['title'], style={'display': 'block', 'marginBottom': '5px'}),
                        html.P(insight['content'], style={'fontSize': '14px', 'marginBottom': '10px'})
                    ],
                    className="mb-3"
                )
            )
        
        # Create anomaly insights display
        anomaly_content = []
        for insight in anomaly_insights[:3]:
            anomaly_content.append(
                html.Div(
                    [
                        html.Strong(insight['title'], style={'display': 'block', 'marginBottom': '5px'}),
                        html.P(insight['content'], style={'fontSize': '14px', 'marginBottom': '10px'})
                    ],
                    className="mb-3"
                )
            )
        
        # Create recommendations display
        recommendations_content = []
        for insight in recommendation_insights[:3]:
            recommendations_content.append(
                html.Div(
                    [
                        html.Strong(insight['title'], style={'display': 'block', 'marginBottom': '5px'}),
                        html.P(insight['content'], style={'fontSize': '14px', 'marginBottom': '10px'})
                    ],
                    className="mb-3"
                )
            )
        
        return (
            exec_summary,
            data_quality_content,
            clustering_content,
            anomaly_content,
            recommendations_content,
            insights
        )
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        error_content = create_data_quality_alert('error', 'Insights Generation Error', str(e))
        return error_content, "", "", "", "", []

@app.callback(
    Output("download-data", "data"),
    [Input("export-data", "n_clicks"),
     Input("download-log", "n_clicks")],
    [State('export-options', 'value'),
     State('export-format', 'value'),
     State('export-filename', 'value')]
)
def export_data(export_clicks, log_clicks, export_options, export_format, filename):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == 'export-data' and export_clicks:
        try:
            # Prepare export data
            include_data = 'processed_data' in export_options
            include_clusters = 'clustering_results' in export_options
            include_anomalies = 'anomaly_scores' in export_options
            include_embeddings = 'pca_embeddings' in export_options
            include_insights = 'analysis_insights' in export_options
            
            # Export results
            file_content, file_name = data_manager.export_results(
                export_format=export_format,
                include_data=include_data,
                include_clusters=include_clusters,
                include_anomalies=include_anomalies,
                include_embeddings=include_embeddings
            )
            
            if export_format == 'excel':
                return dcc.send_bytes(file_content, file_name)
            elif export_format == 'csv':
                return dcc.send_bytes(file_content, file_name)
            elif export_format == 'json':
                return dict(content=file_content.decode(), filename=file_name)
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return dict(content=f"Export error: {str(e)}", filename="error.txt")
    
    elif triggered_id == 'download-log' and log_clicks:
        try:
            log_df = pd.DataFrame(data_manager.pipeline_log)
            return dcc.send_data_frame(
                log_df.to_csv,
                f"pipeline_log_{data_manager.session_id}.csv"
            )
        except Exception as e:
            logger.error(f"Log download failed: {e}")
            return dict(content=f"Log download error: {str(e)}", filename="error.txt")
    
    return dash.no_update

@app.callback(
    [Output('hidden-div', 'children'),
     Output('session-store', 'data', allow_duplicate=True)],
    [Input('clear-session', 'n_clicks')],
    prevent_initial_call=True
)
def clear_session(n_clicks):
    if n_clicks:
        data_manager.clear_session()
        return "Session cleared successfully", {'session_cleared': True}
    return "", dash.no_update

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == '__main__':
    for directory in ['data', 'results', 'logs', 'exports']:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("Starting Enterprise Bioinformatics Analytics Platform")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_ui=False
    )
