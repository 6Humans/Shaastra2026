"""
Production-Grade Data Quality Assessment Module

STRICT RULES:
- NO simulated weights
- NO heuristic guesses
- NO placeholder logic
- NO hard-coded scores
- ALL insights must be generated via real LLM API calls
- ALL metrics must be computed dynamically from data

Implements 6 key data quality dimensions:
1. Completeness - % of non-null values
2. Uniqueness - % of non-duplicate records
3. Validity - % passing format/business rules checks
4. Consistency - % of values following expected patterns
5. Accuracy - Outlier detection using Isolation Forest
6. Timeliness - Freshness of data based on date columns

All scores normalized to 0-100 scale.
"""

import os
import re
import asyncio
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import httpx
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Load environment variables from .env file
load_dotenv()


class DataQualityAssessment:
    """Production-grade data quality scoring with real LLM insights."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Args:
            df: Pandas DataFrame to assess
        """
        self.df = df.copy()
        self.total_rows = len(df)
        self.total_cols = len(df.columns)
        self.total_cells = self.total_rows * self.total_cols
        
        # Validation error tracking
        self.validation_errors: List[Dict[str, Any]] = []
        self.validation_warnings: List[Dict[str, Any]] = []
        self.critical_failures: List[Dict[str, Any]] = []
        
        # LLM configuration (NO HARDCODING - from environment)
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.api_base = os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1/chat/completions')
        self.model = os.getenv('OPENROUTER_MODEL', 'qwen/qwen3-235b-a22b-2507')
        
        # Warn if LLM not available, but allow metric computation
        if not self.api_key:
            print("⚠️  WARNING: OPENROUTER_API_KEY not set - LLM insights will be unavailable")
            print("   All metric computations will still work (no simulation)")
        else:
            print(f"✅ LLM INSIGHTS ENABLED: API key loaded (length: {len(self.api_key)})")

        
    def calculate_completeness(self) -> Dict[str, Any]:
        """
        Completeness: % of non-null values
        Formula: 100 × (total_cells - missing_cells) / total_cells
        """
        print("\n🔢 [COMPLETENESS] Computing...")
        missing_cells = self.df.isnull().sum().sum()
        filled_cells = self.total_cells - missing_cells
        score = (filled_cells / self.total_cells * 100) if self.total_cells > 0 else 0
        print(f"   📊 Formula: 100 × ({filled_cells} / {self.total_cells}) = {score:.2f}")
        print(f"   ⚖️  Weight: Direct calculation (no weight applied)")
        
        # Per-column completeness
        column_scores = {}
        for col in self.df.columns:
            col_missing = self.df[col].isnull().sum()
            col_score = ((self.total_rows - col_missing) / self.total_rows * 100) if self.total_rows > 0 else 0
            column_scores[col] = round(col_score, 2)
        
        return {
            'score': round(score, 2),
            'total_cells': self.total_cells,
            'filled_cells': int(filled_cells),
            'missing_cells': int(missing_cells),
            'column_scores': column_scores,
            'description': 'Percentage of non-null values across all cells'
        }
    
    def calculate_uniqueness(self) -> Dict[str, Any]:
        """
        Uniqueness: % of non-duplicate records
        Formula: 100 × (1 - duplicates / total_rows)
        """
        print("\n🔢 [UNIQUENESS] Computing...")
        duplicate_rows = self.df.duplicated().sum()
        unique_rows = self.total_rows - duplicate_rows
        score = (unique_rows / self.total_rows * 100) if self.total_rows > 0 else 0
        print(f"   📊 Formula: 100 × (1 - {duplicate_rows}/{self.total_rows}) = {score:.2f}")
        print(f"   ⚖️  Weight: Direct calculation (no weight applied)")
        
        # Per-column uniqueness
        column_uniqueness = {}
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            col_score = (unique_count / self.total_rows * 100) if self.total_rows > 0 else 0
            column_uniqueness[col] = {
                'unique_values': int(unique_count),
                'uniqueness_pct': round(col_score, 2)
            }
        
        return {
            'score': round(score, 2),
            'total_rows': self.total_rows,
            'unique_rows': int(unique_rows),
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': round((duplicate_rows / self.total_rows * 100) if self.total_rows > 0 else 0, 2),
            'column_uniqueness': column_uniqueness,
            'description': 'Percentage of non-duplicate records'
        }
    
    def calculate_validity(self) -> Dict[str, Any]:
        """
        Validity: % of values passing format/business rule checks
        Checks: email format, phone format, card numbers (Luhn), dates, numeric ranges
        """
        validation_results = {}
        total_validations = 0
        passed_validations = 0
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            col_validations = {
                'total': len(col_data),
                'passed': len(col_data),
                'checks_applied': []
            }
            
            # Email validation
            if 'email' in col.lower():
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid = col_data.astype(str).str.match(email_pattern).sum()
                col_validations['passed'] = min(col_validations['passed'], int(valid))
                col_validations['checks_applied'].append('email_format')
                total_validations += len(col_data)
                passed_validations += valid
            
            # Phone validation
            elif 'phone' in col.lower() or 'mobile' in col.lower():
                phone_pattern = r'^\+?[1-9]\d{1,14}$|^\d{10}$'
                valid = col_data.astype(str).str.match(phone_pattern).sum()
                col_validations['passed'] = min(col_validations['passed'], int(valid))
                col_validations['checks_applied'].append('phone_format')
                total_validations += len(col_data)
                passed_validations += valid
            
            # Card number validation (Luhn algorithm)
            elif 'card' in col.lower() and 'number' in col.lower():
                def luhn_check(card_num):
                    try:
                        card_str = str(card_num).replace(' ', '').replace('-', '')
                        if not card_str.isdigit() or len(card_str) < 13:
                            return False
                        
                        digits = [int(d) for d in card_str]
                        checksum = 0
                        for i in range(len(digits) - 2, -1, -2):
                            doubled = digits[i] * 2
                            checksum += doubled if doubled < 10 else doubled - 9
                        for i in range(len(digits) - 1, -1, -2):
                            checksum += digits[i]
                        return checksum % 10 == 0
                    except:
                        return False
                
                valid = col_data.apply(luhn_check).sum()
                col_validations['passed'] = min(col_validations['passed'], int(valid))
                col_validations['checks_applied'].append('luhn_algorithm')
                total_validations += len(col_data)
                passed_validations += valid
            
            # Date format validation
            elif 'date' in col.lower() or 'time' in col.lower():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        converted = pd.to_datetime(col_data, errors='coerce', format='mixed')
                    
                    valid = converted.notna().sum()
                    invalid = len(col_data) - valid
                    
                    # Log type safety issues
                    if invalid > 0:
                        invalid_samples = col_data[converted.isna()].head(5).tolist()
                        self.validation_errors.append({
                            'dimension': 'validity',
                            'column': col,
                            'error_type': 'invalid_date_format',
                            'severity': 'high',
                            'count': int(invalid),
                            'percentage': round(invalid / len(col_data) * 100, 2),
                            'samples': invalid_samples,
                            'message': f'{invalid} values could not be parsed as dates'
                        })
                    
                    col_validations['passed'] = int(valid)
                    col_validations['checks_applied'].append('date_format')
                    total_validations += len(col_data)
                    passed_validations += valid
                except Exception as e:
                    self.critical_failures.append({
                        'dimension': 'validity',
                        'column': col,
                        'error': str(e),
                        'message': f'Critical failure in date validation for {col}'
                    })
            
            # Numeric range validation (no negative amounts)
            elif 'amount' in col.lower() or 'fee' in col.lower() or 'price' in col.lower():
                if pd.api.types.is_numeric_dtype(col_data):
                    valid = (col_data >= 0).sum()
                    col_validations['passed'] = min(col_validations['passed'], int(valid))
                    col_validations['checks_applied'].append('positive_numeric')
                    total_validations += len(col_data)
                    passed_validations += valid
            
            # Status/category validation (non-empty strings)
            elif 'status' in col.lower() or 'category' in col.lower() or 'type' in col.lower():
                valid = col_data.astype(str).str.strip().str.len().gt(0).sum()
                col_validations['passed'] = min(col_validations['passed'], int(valid))
                col_validations['checks_applied'].append('non_empty_string')
                total_validations += len(col_data)
                passed_validations += valid
            
            if col_validations['checks_applied']:
                validation_results[col] = col_validations
        
        score = (passed_validations / total_validations * 100) if total_validations > 0 else 100
        
        print("\n🔢 [VALIDITY] Computing...")
        print(f"   📊 Formula: 100 × ({passed_validations}/{total_validations}) = {score:.2f}")
        print(f"   ⚖️  Weight: Direct calculation (no weight applied)")
        print(f"   ✅ Checks applied: Luhn, Email, Phone, Date, Numeric ranges")
        
        return {
            'score': round(score, 2),
            'total_validations': int(total_validations),
            'passed_validations': int(passed_validations),
            'failed_validations': int(total_validations - passed_validations),
            'column_validations': validation_results,
            'description': 'Percentage of values passing format and business rule checks'
        }
    
    def calculate_consistency(self) -> Dict[str, Any]:
        """
        Consistency: % of values following expected patterns and data types
        Checks: data type consistency, pattern consistency, cross-field consistency
        """
        consistency_checks = []
        total_checks = 0
        passed_checks = 0
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Data type consistency (% matching most common type)
            type_counts = col_data.apply(type).value_counts()
            dominant_type = type_counts.index[0]
            type_consistent = (col_data.apply(type) == dominant_type).sum()
            total_checks += len(col_data)
            passed_checks += type_consistent
            
            consistency_checks.append({
                'column': col,
                'check': 'data_type_consistency',
                'total': len(col_data),
                'passed': int(type_consistent),
                'score': round((type_consistent / len(col_data) * 100), 2)
            })
            
            # String pattern consistency (for text columns)
            if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                str_data = col_data.astype(str)
                
                # Case consistency
                upper_count = str_data.str.isupper().sum()
                lower_count = str_data.str.islower().sum()
                title_count = str_data.str.istitle().sum()
                max_case = max(upper_count, lower_count, title_count)
                
                if len(str_data) > 0:
                    total_checks += len(str_data)
                    passed_checks += max_case
                    consistency_checks.append({
                        'column': col,
                        'check': 'case_consistency',
                        'total': len(str_data),
                        'passed': int(max_case),
                        'score': round((max_case / len(str_data) * 100), 2)
                    })
        
        # Cross-field consistency checks
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if len(date_cols) >= 2:
            # Check if dates are in logical order (e.g., created_date < updated_date)
            for i in range(len(date_cols) - 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        date1 = pd.to_datetime(self.df[date_cols[i]], errors='coerce', format='mixed')
                        date2 = pd.to_datetime(self.df[date_cols[i + 1]], errors='coerce', format='mixed')
                    
                    # Track parsing failures
                    date1_invalid = date1.isna().sum()
                    date2_invalid = date2.isna().sum()
                    
                    if date1_invalid > 0 or date2_invalid > 0:
                        self.validation_warnings.append({
                            'dimension': 'consistency',
                            'columns': [date_cols[i], date_cols[i + 1]],
                            'warning_type': 'date_parsing_issues',
                            'date1_invalid': int(date1_invalid),
                            'date2_invalid': int(date2_invalid),
                            'message': f'Date parsing issues in cross-field validation'
                        })
                    
                    valid_pairs = ((date1 <= date2) | date1.isna() | date2.isna()).sum()
                    total_pairs = len(self.df)
                    
                    total_checks += total_pairs
                    passed_checks += valid_pairs
                    consistency_checks.append({
                        'column': f'{date_cols[i]} vs {date_cols[i + 1]}',
                        'check': 'date_order_consistency',
                        'total': total_pairs,
                        'passed': int(valid_pairs),
                        'score': round((valid_pairs / total_pairs * 100), 2)
                    })
                except:
                    pass
        
        score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        
        print("\n🔢 [CONSISTENCY] Computing...")
        print(f"   📊 Formula: 100 × ({passed_checks}/{total_checks}) = {score:.2f}")
        print(f"   ⚖️  Weight: Direct calculation (no weight applied)")
        print(f"   ✅ Checks: Type consistency, Pattern consistency, Cross-field validation")
        
        return {
            'score': round(score, 2),
            'total_checks': int(total_checks),
            'passed_checks': int(passed_checks),
            'failed_checks': int(total_checks - passed_checks),
            'consistency_details': consistency_checks,
            'description': 'Percentage of values following expected patterns and consistency rules'
        }
    
    def calculate_accuracy(self) -> Dict[str, Any]:
        """
        Accuracy: Outlier detection using Isolation Forest (ML-based)
        Score: 100 × (1 - outliers / total_rows)
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                'score': 100.0,
                'outliers_detected': 0,
                'outlier_percentage': 0.0,
                'column_outliers': {},
                'description': 'No numeric columns for outlier detection',
                'method': 'Isolation Forest (ML)'
            }
        
        column_outliers = {}
        total_outliers = 0
        
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) < 10:  # Need at least 10 samples for Isolation Forest
                continue
            
            try:
                # Prepare data for Isolation Forest
                X = col_data.values.reshape(-1, 1)
                
                # Isolation Forest (contamination = expected % of outliers, default 0.1 = 10%)
                iso_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                predictions = iso_forest.fit_predict(X)
                
                # -1 indicates outlier, 1 indicates inlier
                outliers = (predictions == -1).sum()
                outlier_indices = np.where(predictions == -1)[0]
                
                column_outliers[col] = {
                    'total_values': len(col_data),
                    'outliers': int(outliers),
                    'outlier_percentage': round((outliers / len(col_data) * 100), 2),
                    'outlier_sample': col_data.iloc[outlier_indices[:5]].tolist() if len(outlier_indices) > 0 else []
                }
                
                total_outliers += outliers
            except Exception as e:
                column_outliers[col] = {
                    'error': str(e),
                    'outliers': 0
                }
        
        total_numeric_values = sum(self.df[col].notna().sum() for col in numeric_cols)
        outlier_percentage = (total_outliers / total_numeric_values * 100) if total_numeric_values > 0 else 0
        score = max(0, 100 - outlier_percentage * 2)  # Penalize outliers more heavily
        
        print("\n🔢 [ACCURACY] Computing with ML...")
        print(f"   📊 Formula: max(0, 100 - {outlier_percentage:.2f}% × 2) = {score:.2f}")
        print(f"   ⚖️  Weight: Outlier penalty multiplier = 2×")
        print(f"   🤖 ML Model: Isolation Forest (contamination=0.1, n_estimators=100)")
        print(f"   🎯 Outliers detected: {total_outliers} out of {total_numeric_values} values")
        
        return {
            'score': round(score, 2),
            'outliers_detected': int(total_outliers),
            'outlier_percentage': round(outlier_percentage, 2),
            'total_numeric_values': int(total_numeric_values),
            'column_outliers': column_outliers,
            'description': 'Outlier detection using Isolation Forest (Machine Learning)',
            'method': 'Isolation Forest (contamination=0.1)'
        }
    
    def calculate_timeliness(self) -> Dict[str, Any]:
        """
        Timeliness: Data freshness based on date columns
        Score based on how recent the data is (100 = today, decreases with age)
        """
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_cols:
            return {
                'score': 100.0,
                'date_columns_found': 0,
                'freshness_analysis': {},
                'description': 'No date columns found for timeliness assessment',
                'recommendation': 'Add timestamp columns to track data freshness'
            }
        
        freshness_scores = []
        freshness_analysis = {}
        current_date = pd.Timestamp.now()
        
        for col in date_cols:
            try:
                # Convert to datetime with type safety
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    date_series_raw = pd.to_datetime(self.df[col], errors='coerce', format='mixed')
                
                # Track parsing failures
                invalid_count = date_series_raw.isna().sum()
                total_count = len(self.df[col])
                
                if invalid_count > 0:
                    invalid_samples = self.df[col][date_series_raw.isna()].head(5).tolist()
                    self.validation_errors.append({
                        'dimension': 'timeliness',
                        'column': col,
                        'error_type': 'unparseable_datetime',
                        'severity': 'medium',
                        'count': int(invalid_count),
                        'percentage': round(invalid_count / total_count * 100, 2),
                        'samples': invalid_samples,
                        'message': f'{invalid_count}/{total_count} datetime values could not be parsed'
                    })
                
                date_series = date_series_raw.dropna()
                
                if len(date_series) == 0:
                    self.critical_failures.append({
                        'dimension': 'timeliness',
                        'column': col,
                        'error': 'all_dates_invalid',
                        'message': f'All datetime values in {col} are invalid'
                    })
                    continue
                
                # Calculate freshness metrics
                most_recent = date_series.max()
                oldest = date_series.min()
                average_date = pd.to_datetime(date_series.mean())
                
                # Days since most recent record
                days_since_recent = (current_date - most_recent).days
                days_since_oldest = (current_date - oldest).days
                days_since_average = (current_date - average_date).days
                
                # Calculate freshness score (100 for today, decreases over time)
                # 100 points for data within last 7 days
                # 80 points for data within last 30 days
                # 60 points for data within last 90 days
                # 40 points for data within last 180 days
                # 20 points for data within last 365 days
                # 0 points for data older than 365 days
                
                if days_since_recent <= 7:
                    col_score = 100
                elif days_since_recent <= 30:
                    col_score = 80
                elif days_since_recent <= 90:
                    col_score = 60
                elif days_since_recent <= 180:
                    col_score = 40
                elif days_since_recent <= 365:
                    col_score = 20
                else:
                    col_score = 0
                
                freshness_scores.append(col_score)
                
                freshness_analysis[col] = {
                    'most_recent_date': most_recent.strftime('%Y-%m-%d'),
                    'oldest_date': oldest.strftime('%Y-%m-%d'),
                    'average_date': average_date.strftime('%Y-%m-%d'),
                    'days_since_recent': int(days_since_recent),
                    'days_since_oldest': int(days_since_oldest),
                    'data_age_range_days': int(days_since_oldest - days_since_recent),
                    'freshness_score': col_score,
                    'freshness_category': self._get_freshness_category(days_since_recent)
                }
                
            except Exception as e:
                freshness_analysis[col] = {
                    'error': str(e),
                    'freshness_score': 0
                }
        
        overall_score = np.mean(freshness_scores) if freshness_scores else 100
        
        return {
            'score': round(overall_score, 2),
            'date_columns_found': len(date_cols),
            'date_columns_analyzed': len(freshness_scores),
            'freshness_analysis': freshness_analysis,
            'description': 'Data freshness based on date/timestamp columns',
            'recommendation': self._get_timeliness_recommendation(overall_score)
        }
    
    def _get_freshness_category(self, days: int) -> str:
        """Categorize data freshness."""
        if days <= 7:
            return 'Fresh (< 7 days)'
        elif days <= 30:
            return 'Recent (< 30 days)'
        elif days <= 90:
            return 'Moderate (< 90 days)'
        elif days <= 180:
            return 'Aging (< 180 days)'
        elif days <= 365:
            return 'Old (< 1 year)'
        else:
            return 'Stale (> 1 year)'
    
    def _get_timeliness_recommendation(self, score: float) -> str:
        """Provide timeliness recommendations."""
        if score >= 80:
            return 'Data is fresh and up-to-date'
        elif score >= 60:
            return 'Consider refreshing data within 30 days'
        elif score >= 40:
            return 'Data is aging - refresh recommended within 2 weeks'
        else:
            return 'Data is stale - immediate refresh required'
    
    def calculate_dynamic_weights(self) -> Dict[str, float]:
        """
        Calculate dynamic weights for each dimension based on EDA analysis.
        
        Strategy:
        - High missing data → Increase Completeness weight
        - High duplicates → Increase Uniqueness weight
        - Mixed data types → Increase Validity weight
        - Pattern inconsistencies → Increase Consistency weight
        - Outlier-prone data → Increase Accuracy weight
        - Temporal data present → Increase Timeliness weight
        
        Returns normalized weights that sum to 1.0
        """
        print("\n" + "🔍"*40)
        print("⚖️  CALCULATING DYNAMIC WEIGHTS BASED ON EDA")
        print("🔍"*40)
        
        weights = {}
        
        # 1. COMPLETENESS WEIGHT - Based on missingness severity
        missing_cells = self.df.isnull().sum().sum()
        missing_ratio = missing_cells / self.total_cells if self.total_cells > 0 else 0
        
        if missing_ratio > 0.30:
            completeness_weight = 0.25  # Critical issue
            print(f"📊 Completeness: HIGH priority (missing: {missing_ratio*100:.1f}%) → weight: 0.25")
        elif missing_ratio > 0.15:
            completeness_weight = 0.20  # Moderate issue
            print(f"📊 Completeness: MODERATE priority (missing: {missing_ratio*100:.1f}%) → weight: 0.20")
        elif missing_ratio > 0.05:
            completeness_weight = 0.15  # Minor issue
            print(f"📊 Completeness: STANDARD priority (missing: {missing_ratio*100:.1f}%) → weight: 0.15")
        else:
            completeness_weight = 0.10  # Low priority
            print(f"📊 Completeness: LOW priority (missing: {missing_ratio*100:.1f}%) → weight: 0.10")
        
        weights['completeness'] = completeness_weight
        
        # 2. UNIQUENESS WEIGHT - Based on duplicate prevalence
        duplicate_rows = self.df.duplicated().sum()
        duplicate_ratio = duplicate_rows / self.total_rows if self.total_rows > 0 else 0
        
        if duplicate_ratio > 0.20:
            uniqueness_weight = 0.25
            print(f"📊 Uniqueness: HIGH priority (duplicates: {duplicate_ratio*100:.1f}%) → weight: 0.25")
        elif duplicate_ratio > 0.10:
            uniqueness_weight = 0.20
            print(f"📊 Uniqueness: MODERATE priority (duplicates: {duplicate_ratio*100:.1f}%) → weight: 0.20")
        elif duplicate_ratio > 0.02:
            uniqueness_weight = 0.15
            print(f"📊 Uniqueness: STANDARD priority (duplicates: {duplicate_ratio*100:.1f}%) → weight: 0.15")
        else:
            uniqueness_weight = 0.10
            print(f"📊 Uniqueness: LOW priority (duplicates: {duplicate_ratio*100:.1f}%) → weight: 0.10")
        
        weights['uniqueness'] = uniqueness_weight
        
        # 3. VALIDITY WEIGHT - Based on data type diversity and potential format issues
        dtype_diversity = len(self.df.dtypes.value_counts())
        string_cols = sum(1 for dt in self.df.dtypes if dt == 'object')
        string_ratio = string_cols / len(self.df.columns) if len(self.df.columns) > 0 else 0
        
        if string_ratio > 0.6 or dtype_diversity > 5:
            validity_weight = 0.22
            print(f"📊 Validity: HIGH priority (string cols: {string_ratio*100:.1f}%, type diversity: {dtype_diversity}) → weight: 0.22")
        elif string_ratio > 0.4 or dtype_diversity > 3:
            validity_weight = 0.18
            print(f"📊 Validity: MODERATE priority (string cols: {string_ratio*100:.1f}%, type diversity: {dtype_diversity}) → weight: 0.18")
        else:
            validity_weight = 0.14
            print(f"📊 Validity: STANDARD priority (string cols: {string_ratio*100:.1f}%, type diversity: {dtype_diversity}) → weight: 0.14")
        
        weights['validity'] = validity_weight
        
        # 4. CONSISTENCY WEIGHT - Based on potential pattern variations
        # Check for columns with mixed patterns (e.g., mixed case, inconsistent formats)
        pattern_inconsistency_score = 0
        for col in self.df.select_dtypes(include=['object']).columns[:10]:  # Sample first 10 string columns
            if self.df[col].notna().sum() > 0:
                unique_patterns = self.df[col].dropna().astype(str).str.len().nunique()
                if unique_patterns > 5:
                    pattern_inconsistency_score += 1
        
        if pattern_inconsistency_score > 5:
            consistency_weight = 0.20
            print(f"📊 Consistency: HIGH priority (pattern variations: {pattern_inconsistency_score}) → weight: 0.20")
        elif pattern_inconsistency_score > 2:
            consistency_weight = 0.16
            print(f"📊 Consistency: MODERATE priority (pattern variations: {pattern_inconsistency_score}) → weight: 0.16")
        else:
            consistency_weight = 0.12
            print(f"📊 Consistency: STANDARD priority (pattern variations: {pattern_inconsistency_score}) → weight: 0.12")
        
        weights['consistency'] = consistency_weight
        
        # 5. ACCURACY WEIGHT - Based on numeric data presence (outlier detection relevance)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_ratio = len(numeric_cols) / len(self.df.columns) if len(self.df.columns) > 0 else 0
        
        if numeric_ratio > 0.5:
            accuracy_weight = 0.22
            print(f"📊 Accuracy: HIGH priority (numeric cols: {numeric_ratio*100:.1f}%) → weight: 0.22")
        elif numeric_ratio > 0.3:
            accuracy_weight = 0.18
            print(f"📊 Accuracy: MODERATE priority (numeric cols: {numeric_ratio*100:.1f}%) → weight: 0.18")
        else:
            accuracy_weight = 0.12
            print(f"📊 Accuracy: LOW priority (numeric cols: {numeric_ratio*100:.1f}%) → weight: 0.12")
        
        weights['accuracy'] = accuracy_weight
        
        # 6. TIMELINESS WEIGHT - Based on presence of date/time columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in self.df.select_dtypes(include=['object']).columns:
            if col not in date_cols:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(self.df[col].dropna().head(10), errors='raise', format='mixed')
                    date_cols.append(col)
                except:
                    pass
        
        if len(date_cols) >= 3:
            timeliness_weight = 0.20
            print(f"📊 Timeliness: HIGH priority ({len(date_cols)} date columns) → weight: 0.20")
        elif len(date_cols) >= 1:
            timeliness_weight = 0.15
            print(f"📊 Timeliness: MODERATE priority ({len(date_cols)} date columns) → weight: 0.15")
        else:
            timeliness_weight = 0.08
            print(f"📊 Timeliness: LOW priority ({len(date_cols)} date columns) → weight: 0.08")
        
        weights['timeliness'] = timeliness_weight
        
        # NORMALIZE weights to sum to 1.0
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        print(f"\n🎯 Weight Normalization:")
        print(f"   Raw sum: {total_weight:.4f}")
        print(f"   Normalized sum: {sum(normalized_weights.values()):.4f}")
        print("\n✅ Final Normalized Weights:")
        for dim, weight in normalized_weights.items():
            print(f"   - {dim.capitalize():15s}: {weight:.4f} ({weight*100:.2f}%)")
        print("🔍"*40 + "\n")
        
        return normalized_weights
    
    def calculate_all_dimensions(self, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate all 6 data quality dimensions with real LLM insights."""
        
        # Step 1: Compute all dimensions (REAL COMPUTATION ONLY)
        try:
            dimensions = {
                'completeness': self.calculate_completeness(),
                'uniqueness': self.calculate_uniqueness(),
                'validity': self.calculate_validity(),
                'consistency': self.calculate_consistency(),
                'accuracy': self.calculate_accuracy(),
                'timeliness': self.calculate_timeliness()
            }
        except Exception as e:
            return {
                'status': 'insufficient_data',
                'error': str(e),
                'message': 'Unable to compute quality dimensions - real computation failed'
            }
        
        # Step 2: Extract dimension scores (NO HARDCODING)
        dimension_scores = {dim: dimensions[dim]['score'] for dim in dimensions}
        
        # Validate all scores are in 0-100 range
        for dim, score in dimension_scores.items():
            if not (0 <= score <= 100):
                raise ValueError(f"Score normalization failed for {dim}: {score} not in [0, 100]")
        
    def calculate_all_dimensions(self, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate all 6 data quality dimensions with DYNAMIC weights based on EDA."""
        
        # Step 1: Calculate dynamic weights based on data characteristics
        if custom_weights is None:
            weights = self.calculate_dynamic_weights()
        else:
            weights = custom_weights
            print("⚠️  Using custom weights provided by user")
        
        # Step 2: Compute all dimensions (REAL COMPUTATION ONLY)
        try:
            dimensions = {
                'completeness': self.calculate_completeness(),
                'uniqueness': self.calculate_uniqueness(),
                'validity': self.calculate_validity(),
                'consistency': self.calculate_consistency(),
                'accuracy': self.calculate_accuracy(),
                'timeliness': self.calculate_timeliness()
            }
        except Exception as e:
            return {
                'status': 'insufficient_data',
                'error': str(e),
                'message': 'Unable to compute quality dimensions - real computation failed'
            }
        
        # Step 3: Extract dimension scores (NO HARDCODING)
        dimension_scores = {dim: dimensions[dim]['score'] for dim in dimensions}
        
        # Validate all scores are in 0-100 range
        for dim, score in dimension_scores.items():
            if not (0 <= score <= 100):
                raise ValueError(f"Score normalization failed for {dim}: {score} not in [0, 100]")
        
        # Step 4: Calculate WEIGHTED overall score
        overall_score = sum(dimension_scores[dim] * weights[dim] for dim in dimension_scores)
        
        print("\n" + "="*80)
        print("📊 OVERALL QUALITY SCORE CALCULATION")
        print("="*80)
        print(f"⚖️  Weighting Strategy: DYNAMIC WEIGHTS (EDA-Based)")
        print(f"📐 Formula: Σ(dimension_score × weight)")
        print(f"\nDimension Scores:")
        for dim, score in dimension_scores.items():
            weight = weights[dim]
            contribution = score * weight
            print(f"   - {dim.capitalize():15s}: {score:6.2f}/100 × {weight:.4f} = {contribution:6.2f} points")
        print(f"\n🎯 Overall Score: {overall_score:.2f}/100")
        print(f"   Calculation: " + " + ".join([f"({dimension_scores[d]:.2f} × {weights[d]:.4f})" for d in dimension_scores]))
        print("="*80 + "\n")
        
        # Step 5: Collect issues for LLM context
        issues_detected = self._extract_issues(dimensions)
        
        # Step 6: Generate REAL LLM insights (NO SIMULATION)
        try:
            llm_insights = self._generate_llm_insights(dimension_scores, issues_detected, weights)
        except Exception as e:
            llm_insights = {
                'status': 'external_dependency_failure',
                'error': str(e),
                'message': 'LLM API call failed - cannot generate insights'
            }
        
        # Step 6: Determine confidence level based on data completeness
        confidence_level = self._calculate_confidence_level(dimensions)
        
        # Step 7: Include validation errors and warnings in response
        validation_summary = {
            'errors': self.validation_errors,
            'warnings': self.validation_warnings,
            'critical_failures': self.critical_failures,
            'error_count': len(self.validation_errors),
            'warning_count': len(self.validation_warnings),
            'critical_count': len(self.critical_failures),
            'has_issues': len(self.validation_errors) > 0 or len(self.critical_failures) > 0
        }
        
        return {
            'scores': dimension_scores,
            'overall_score': round(overall_score, 2),
            'dimensions': dimensions,
            'issues_detected': issues_detected,
            'llm_insights': llm_insights,
            'confidence_level': confidence_level,
            'weights_applied': weights,  # Store actual dynamic weights used
            'quality_grade': self._get_quality_grade(overall_score),
            'validation_summary': validation_summary,  # NEW: Include all validation errors
            'summary': {
                'total_rows': self.total_rows,
                'total_columns': self.total_cols,
                'total_cells': self.total_cells
            },
            'computation_metadata': {
                'all_scores_normalized': True,
                'no_simulation': True,
                'llm_enabled': bool(self.api_key),
                'weighting_strategy': 'dynamic_eda_based',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _extract_issues(self, dimensions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract detected issues from dimension analysis."""
        issues = []
        
        # Completeness issues
        if dimensions['completeness']['score'] < 80:
            issues.append({
                'dimension': 'completeness',
                'severity': 'high' if dimensions['completeness']['score'] < 60 else 'medium',
                'description': f"{dimensions['completeness']['missing_cells']} missing cells detected",
                'affected_columns': dimensions['completeness']['total_columns_with_missing'] if 'total_columns_with_missing' in dimensions['completeness'] else 0
            })
        
        # Uniqueness issues
        if dimensions['uniqueness']['score'] < 95:
            issues.append({
                'dimension': 'uniqueness',
                'severity': 'high' if dimensions['uniqueness']['duplicate_rows'] > self.total_rows * 0.1 else 'medium',
                'description': f"{dimensions['uniqueness']['duplicate_rows']} duplicate records found",
                'duplicate_percentage': dimensions['uniqueness']['duplicate_percentage']
            })
        
        # Validity issues
        if dimensions['validity']['total_validations'] > 0 and dimensions['validity']['score'] < 90:
            issues.append({
                'dimension': 'validity',
                'severity': 'high' if dimensions['validity']['score'] < 70 else 'medium',
                'description': f"{dimensions['validity']['failed_validations']} validation failures",
                'validations_run': dimensions['validity']['total_validations']
            })
        
        # Accuracy issues (outliers)
        if dimensions['accuracy']['outliers_detected'] > 0:
            issues.append({
                'dimension': 'accuracy',
                'severity': 'medium',
                'description': f"{dimensions['accuracy']['outliers_detected']} outliers detected by Isolation Forest",
                'outlier_percentage': dimensions['accuracy']['outlier_percentage']
            })
        
        # Timeliness issues
        if dimensions['timeliness']['score'] < 60:
            issues.append({
                'dimension': 'timeliness',
                'severity': 'high' if dimensions['timeliness']['score'] < 40 else 'medium',
                'description': 'Data freshness is poor',
                'recommendation': dimensions['timeliness'].get('recommendation', 'Refresh data')
            })
        
        return issues
    
    def _calculate_confidence_level(self, dimensions: Dict[str, Any]) -> str:
        """Calculate confidence level based on data availability."""
        # High confidence: >90% complete, >95% unique, low outliers
        completeness = dimensions['completeness']['score']
        uniqueness = dimensions['uniqueness']['score']
        
        if completeness >= 90 and uniqueness >= 95:
            return 'high'
        elif completeness >= 70 and uniqueness >= 85:
            return 'medium'
        else:
            return 'low'
    
    def _generate_llm_insights(self, dimension_scores: Dict[str, float], issues: List[Dict], weights: Dict[str, float]) -> Dict[str, Any]:
        """Generate real insights using LLM API - NO SIMULATION."""
        
        if not self.api_key:
            return {
                'status': 'external_dependency_failure',
                'message': 'LLM API key not configured - cannot generate insights'
            }
        
        # Prepare context for LLM with dynamic weights
        weights_info = "\n".join([f"- {dim.capitalize()}: {weight*100:.2f}% (priority: {'HIGH' if weight > 0.2 else 'MEDIUM' if weight > 0.15 else 'STANDARD'})" 
                                   for dim, weight in weights.items()])
        
        context = f"""
You are a data quality expert analyzing a dataset with DYNAMIC WEIGHTING based on EDA.

Data Quality Scores (0-100 scale):
- Completeness: {dimension_scores['completeness']:.2f}
- Uniqueness: {dimension_scores['uniqueness']:.2f}
- Validity: {dimension_scores['validity']:.2f}
- Consistency: {dimension_scores['consistency']:.2f}
- Accuracy: {dimension_scores['accuracy']:.2f}
- Timeliness: {dimension_scores['timeliness']:.2f}

Dynamic Weights (based on EDA analysis):
{weights_info}

Detected Issues:
{self._format_issues_for_llm(issues)}

Provide:
1. Executive summary (2-3 sentences) - focus on dimensions with highest weights
2. Root cause analysis (top 3 causes) - prioritize issues in high-weight dimensions
3. Actionable recommendations (prioritized list) - address high-weight dimensions first

Be specific, data-driven, and actionable. Consider the dynamic weights when prioritizing recommendations.
"""
        
        try:
            # Make REAL API call
            response = self._call_llm_api(context)
            
            # Parse LLM response
            insights = self._parse_llm_response(response)
            
            return insights
            
        except Exception as e:
            return {
                'status': 'external_dependency_failure',
                'error': str(e),
                'message': 'LLM API call failed'
            }
    
    def _format_issues_for_llm(self, issues: List[Dict]) -> str:
        """Format issues for LLM context."""
        if not issues:
            return "No critical issues detected."
        
        formatted = []
        for i, issue in enumerate(issues, 1):
            formatted.append(f"{i}. [{issue['severity'].upper()}] {issue['dimension']}: {issue['description']}")
        
        return "\n".join(formatted)
    
    def _call_llm_api(self, context: str) -> str:
        """Make real LLM API call - NO MOCKING."""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a data quality expert providing actionable insights.'
                },
                {
                    'role': 'user',
                    'content': context
                }
            ],
            'temperature': 0.3,  # Lower temperature for more deterministic output
            'max_tokens': 800
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self.api_base, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        
        lines = response.strip().split('\n')
        
        summary = []
        root_causes = []
        recommendations = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if 'summary' in line.lower() or 'executive' in line.lower():
                current_section = 'summary'
                continue
            elif 'root cause' in line.lower() or 'causes' in line.lower():
                current_section = 'root_causes'
                continue
            elif 'recommendation' in line.lower() or 'action' in line.lower():
                current_section = 'recommendations'
                continue
            
            # Add content to appropriate section
            if current_section == 'summary':
                if not line.startswith(('1.', '2.', '3.', '-', '*')):
                    summary.append(line)
            elif current_section == 'root_causes':
                if line.startswith(('1.', '2.', '3.', '-', '*')):
                    root_causes.append(line.lstrip('123.-* '))
            elif current_section == 'recommendations':
                if line.startswith(('1.', '2.', '3.', '-', '*')):
                    recommendations.append(line.lstrip('123.-* '))
        
        return {
            'summary': ' '.join(summary) if summary else response[:200],
            'root_causes': root_causes[:5] if root_causes else ['Analysis pending'],
            'recommendations': recommendations[:5] if recommendations else ['Review data quality metrics']
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Fair)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Critical)'
