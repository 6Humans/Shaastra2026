"""
FastAPI endpoint for CSV-based transaction analysis.

Upload CSV files and receive comprehensive EDA analysis in both JSON and Markdown formats.
"""

import asyncio
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.record_processor import RecordOrchestrator, Record
from src.data_quality import DataQualityAssessment

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Bank Transaction Analysis API",
    description="Upload CSV files for AI-powered transaction analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sanitize_for_json(obj):
    """
    Recursively sanitize data to be JSON-compliant.
    Replaces NaN, Infinity, -Infinity with None.
    """
    import math
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def perform_eda(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Perform comprehensive EDA with automated data quality assessment."""
    eda_results = {}
    
    # Initialize Data Quality Assessment
    dq_assessment = DataQualityAssessment(df)
    
    # Calculate all 6 data quality dimensions
    quality_results = dq_assessment.calculate_all_dimensions()
    eda_results['data_quality_dimensions'] = quality_results
    
    # 1. Dataset Shape
    eda_results['shape'] = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1])
    }
    
    # 2. Column Information
    eda_results['column_types'] = {
        'numeric': len(df.select_dtypes(include=[np.number]).columns),
        'object': len(df.select_dtypes(include=['object']).columns),
        'datetime': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    # 3. Missing Values Analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_data = pd.DataFrame({
        'count': missing,
        'percentage': missing_pct
    })
    missing_data = missing_data[missing_data['count'] > 0].sort_values('count', ascending=False)
    
    eda_results['missing_values'] = {
        'total_columns_with_missing': int(len(missing_data)),
        'columns': {
            col: {
                'count': int(row['count']),
                'percentage': float(row['percentage'])
            }
            for col, row in missing_data.head(20).iterrows()
        }
    }
    
    # 4. Data Quality Score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness_pct = ((total_cells - missing_cells) / total_cells * 100).round(2)
    
    eda_results['data_quality'] = {
        'completeness_percentage': float(completeness_pct),
        'total_cells': int(total_cells),
        'missing_cells': int(missing_cells)
    }
    
    # 5. Duplicates
    duplicates = df.duplicated().sum()
    eda_results['duplicates'] = {
        'count': int(duplicates),
        'percentage': float((duplicates / len(df) * 100).round(2))
    }
    
    # 6. Numeric Statistics (summary)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        numeric_summary = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            numeric_summary[col] = {
                'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                'median': float(df[col].median()) if pd.notna(df[col].median()) else None,
                'std': float(df[col].std()) if pd.notna(df[col].std()) else None,
                'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                'max': float(df[col].max()) if pd.notna(df[col].max()) else None
            }
        eda_results['numeric_summary'] = numeric_summary
    
    return eda_results


def create_anomaly_report(eda_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive anomaly report consolidating all anomaly detections.
    
    Returns:
        Dictionary with categorized anomaly information
    """
    dq = eda_results.get('data_quality_dimensions', {})
    dimensions = dq.get('dimensions', {})
    validation_summary = dq.get('validation_summary', {})
    
    anomaly_report = {
        "summary": {
            "total_anomalies": 0,
            "critical_count": validation_summary.get('critical_count', 0),
            "error_count": validation_summary.get('error_count', 0),
            "warning_count": validation_summary.get('warning_count', 0),
            "outlier_count": dimensions.get('accuracy', {}).get('outliers_detected', 0),
            "has_critical_issues": validation_summary.get('critical_count', 0) > 0,
            "overall_anomaly_rate": 0.0
        },
        
        "ml_detected_outliers": {
            "total_outliers": dimensions.get('accuracy', {}).get('outliers_detected', 0),
            "outlier_percentage": dimensions.get('accuracy', {}).get('outlier_percentage', 0),
            "detection_method": "Isolation Forest (contamination=0.1)",
            "by_column": []
        },
        
        "type_validation_errors": {
            "total_errors": validation_summary.get('error_count', 0),
            "errors": []
        },
        
        "consistency_warnings": {
            "total_warnings": validation_summary.get('warning_count', 0),
            "warnings": []
        },
        
        "critical_failures": {
            "total_failures": validation_summary.get('critical_count', 0),
            "failures": []
        },
        
        "high_risk_columns": []
    }
    
    # Extract ML-detected outliers by column
    column_outliers = dimensions.get('accuracy', {}).get('column_outliers', {})
    for col, outlier_info in column_outliers.items():
        if outlier_info.get('outliers', 0) > 0:
            anomaly_report['ml_detected_outliers']['by_column'].append({
                "column": col,
                "outlier_count": outlier_info.get('outliers', 0),
                "outlier_percentage": outlier_info.get('outlier_percentage', 0),
                "sample_values": outlier_info.get('outlier_sample', [])[:5],
                "severity": "high" if outlier_info.get('outlier_percentage', 0) > 10 else "medium"
            })
    
    # Extract validation errors
    for error in validation_summary.get('errors', []):
        anomaly_report['type_validation_errors']['errors'].append({
            "column": error.get('column'),
            "error_type": error.get('error_type'),
            "severity": error.get('severity'),
            "count": error.get('count'),
            "percentage": error.get('percentage'),
            "sample_invalid_values": error.get('samples', [])[:3],
            "message": error.get('message'),
            "dimension": error.get('dimension')
        })
    
    # Extract warnings
    for warning in validation_summary.get('warnings', []):
        anomaly_report['consistency_warnings']['warnings'].append({
            "warning_type": warning.get('warning_type'),
            "affected_columns": warning.get('columns', []),
            "message": warning.get('message'),
            "dimension": warning.get('dimension'),
            "details": {
                k: v for k, v in warning.items() 
                if k not in ['warning_type', 'columns', 'message', 'dimension']
            }
        })
    
    # Extract critical failures
    for failure in validation_summary.get('critical_failures', []):
        anomaly_report['critical_failures']['failures'].append({
            "column": failure.get('column'),
            "error": failure.get('error'),
            "message": failure.get('message'),
            "dimension": failure.get('dimension')
        })
    
    # Identify high-risk columns (multiple anomaly types)
    column_risk_score = {}
    
    # Score from outliers
    for outlier in anomaly_report['ml_detected_outliers']['by_column']:
        col = outlier['column']
        column_risk_score[col] = column_risk_score.get(col, 0) + outlier['outlier_percentage']
    
    # Score from validation errors
    for error in anomaly_report['type_validation_errors']['errors']:
        col = error['column']
        severity_weight = {'high': 50, 'medium': 25, 'low': 10}.get(error['severity'], 10)
        column_risk_score[col] = column_risk_score.get(col, 0) + severity_weight
    
    # Score from critical failures
    for failure in anomaly_report['critical_failures']['failures']:
        col = failure.get('column')
        if col:
            column_risk_score[col] = column_risk_score.get(col, 0) + 100
    
    # Sort and create high-risk list
    sorted_risks = sorted(column_risk_score.items(), key=lambda x: x[1], reverse=True)
    for col, risk_score in sorted_risks[:10]:  # Top 10 risky columns
        anomaly_types = []
        
        # Check if has outliers
        if any(o['column'] == col for o in anomaly_report['ml_detected_outliers']['by_column']):
            outlier_data = next(o for o in anomaly_report['ml_detected_outliers']['by_column'] if o['column'] == col)
            anomaly_types.append(f"outliers ({outlier_data['outlier_percentage']:.2f}%)")
        
        # Check if has validation errors
        if any(e['column'] == col for e in anomaly_report['type_validation_errors']['errors']):
            error_data = [e for e in anomaly_report['type_validation_errors']['errors'] if e['column'] == col]
            anomaly_types.extend([f"{e['error_type']} ({e['severity']})" for e in error_data])
        
        # Check if has critical failures
        if any(f.get('column') == col for f in anomaly_report['critical_failures']['failures']):
            anomaly_types.append("critical_failure")
        
        anomaly_report['high_risk_columns'].append({
            "column": col,
            "risk_score": round(risk_score, 2),
            "anomaly_types": anomaly_types,
            "severity": "critical" if risk_score > 100 else "high" if risk_score > 50 else "medium"
        })
    
    # Calculate totals
    total_anomalies = (
        anomaly_report['ml_detected_outliers']['total_outliers'] +
        anomaly_report['type_validation_errors']['total_errors'] +
        anomaly_report['critical_failures']['total_failures']
    )
    
    total_data_points = eda_results.get('shape', {}).get('rows', 1) * eda_results.get('shape', {}).get('columns', 1)
    anomaly_rate = (total_anomalies / total_data_points * 100) if total_data_points > 0 else 0
    
    anomaly_report['summary']['total_anomalies'] = total_anomalies
    anomaly_report['summary']['overall_anomaly_rate'] = round(anomaly_rate, 2)
    
    return anomaly_report


def create_records_from_csv(df: pd.DataFrame, dataset_name: str, num_samples: int = 5) -> list[Record]:
    """Create Record objects from CSV data for AI analysis."""
    records = []
    
    # Sample records from the dataset
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    for idx, row in sample_df.iterrows():
        # Convert row to dictionary, handling NaN values
        row_dict = row.to_dict()
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
        
        record = Record(
            record_id=f"{dataset_name}-{idx}",
            data=row_dict,
            metadata={
                "source": "csv_upload",
                "dataset": dataset_name,
                "upload_time": datetime.now().isoformat()
            }
        )
        records.append(record)
    
    return records


async def process_records_with_agents(records: list[Record], eda_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Process records through AI agent system with EDA context."""
    orchestrator = RecordOrchestrator(eda_context=eda_context)
    results = await orchestrator.process_batch(records)
    
    # Aggregate results
    total_records = len(results)
    successful = sum(1 for r in results if r.status.value == "completed")
    failed = total_records - successful
    
    # Calculate average metrics
    quality_scores = []
    insights = []
    
    for r in results:
        # Extract quality score from scoring output
        if r.scoring_output and isinstance(r.scoring_output, dict):
            score = r.scoring_output.get("overall_quality_score")
            if score is not None:
                quality_scores.append(score)
        
        # Extract insights from insight output
        if r.insight_output and isinstance(r.insight_output, dict):
            if r.insight_output.get("insights"):
                insights.extend(r.insight_output["insights"])
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Convert RecordResult objects to dictionaries
    detailed_results = []
    for r in results:
        detailed_results.append({
            "record_id": r.record_id,
            "status": r.status.value,
            "quality_score": r.scoring_output.get("overall_quality_score") if r.scoring_output else None,
            "ai_insights": r.insight_output.get("insights", []) if r.insight_output else [],
            "predictions": r.predictive_output if r.predictive_output else None,
            "processing_time_ms": r.processing_time_ms,
            "errors": r.errors
        })
    
    return {
        "summary": {
            "total_records_processed": total_records,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/total_records*100):.2f}%" if total_records > 0 else "0%"
        },
        "metrics": {
            "average_quality_score": round(avg_quality, 3),
            "total_insights_generated": len(insights)
        },
        "detailed_results": detailed_results
    }


def generate_markdown_report(dataset_name: str, eda_results: Dict, ai_analysis: Dict) -> str:
    """Generate comprehensive markdown report with data quality dimensions."""
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    dq = eda_results['data_quality_dimensions']
    
    md = f"""# 📊 Transaction Analysis Report

**Dataset:** {dataset_name}  
**Generated:** {timestamp}  
**Analysis Tool:** AI-Powered EDA with Real LLM Insights (No Simulation)

---

## 🎯 Executive Summary

Analyzed **{eda_results['shape']['rows']:,} transactions** across **{eda_results['shape']['columns']} columns** using production-grade data quality assessment.

**Overall Data Quality Score:** {dq['overall_score']:.2f}/100 ({dq['quality_grade']})

**Confidence Level:** {dq['confidence_level'].upper()}

### 📊 Data Quality Dimensions (All Scores: 0-100 Scale)

| Dimension | Score | Grade | Description |
|-----------|-------|-------|-------------|
| **Completeness** | {dq['scores']['completeness']:.2f} | {_score_to_grade(dq['scores']['completeness'])} | {dq['dimensions']['completeness']['description']} |
| **Uniqueness** | {dq['scores']['uniqueness']:.2f} | {_score_to_grade(dq['scores']['uniqueness'])} | {dq['dimensions']['uniqueness']['description']} |
| **Validity** | {dq['scores']['validity']:.2f} | {_score_to_grade(dq['scores']['validity'])} | {dq['dimensions']['validity']['description']} |
| **Consistency** | {dq['scores']['consistency']:.2f} | {_score_to_grade(dq['scores']['consistency'])} | {dq['dimensions']['consistency']['description']} |
| **Accuracy** | {dq['scores']['accuracy']:.2f} | {_score_to_grade(dq['scores']['accuracy'])} | {dq['dimensions']['accuracy']['description']} |
| **Timeliness** | {dq['scores']['timeliness']:.2f} | {_score_to_grade(dq['scores']['timeliness'])} | {dq['dimensions']['timeliness']['description']} |
| **Integrity** | {dq['scores']['integrity']:.2f} | {_score_to_grade(dq['scores']['integrity'])} | {dq['dimensions']['integrity']['description']} |

---

## 🔍 Issues Detected

**Total Issues:** {len(dq['issues_detected'])}

"""
    
    if dq['issues_detected']:
        for i, issue in enumerate(dq['issues_detected'], 1):
            severity_emoji = '🚨' if issue['severity'] == 'high' else '⚠️'
            md += f"{i}. {severity_emoji} **{issue['dimension'].upper()}** ({issue['severity']}): {issue['description']}\n"
    else:
        md += "✅ **No critical issues detected!**\n"
    
    md += f"""
---

## 🤖 LLM-Generated Insights (Real API Call)

"""
    
    llm_insights = dq.get('llm_insights', {})
    
    if llm_insights.get('status') == 'external_dependency_failure':
        md += f"⚠️ **LLM insights unavailable:** {llm_insights.get('message')}\n\n"
    else:
        md += f"**Summary:**\n{llm_insights.get('summary', 'No summary available')}\n\n"
        
        if llm_insights.get('root_causes'):
            md += "**Root Causes:**\n"
            for i, cause in enumerate(llm_insights['root_causes'], 1):
                md += f"{i}. {cause}\n"
            md += "\n"
        
        if llm_insights.get('recommendations'):
            md += "**Recommendations:**\n"
            for i, rec in enumerate(llm_insights['recommendations'], 1):
                md += f"{i}. {rec}\n"
            md += "\n"
    
    # Add validation errors section
    validation_summary = dq.get('validation_summary', {})
    if validation_summary.get('has_issues', False):
        md += f"""
---

## ⚠️ Validation Errors & Type Safety Issues

"""
        
        # Critical failures
        if validation_summary.get('critical_count', 0) > 0:
            md += "### 🚨 Critical Failures\n\n"
            for failure in validation_summary.get('critical_failures', []):
                md += f"- **{failure['dimension'].upper()}** - Column: `{failure.get('column', 'N/A')}`\n"
                md += f"  - Error: `{failure['error']}`\n"
                md += f"  - Message: {failure['message']}\n\n"
        
        # Validation errors
        if validation_summary.get('error_count', 0) > 0:
            md += "### ❌ Validation Errors\n\n"
            for error in validation_summary.get('errors', []):
                md += f"- **{error['dimension'].upper()}** - Column: `{error['column']}`\n"
                md += f"  - Type: {error['error_type']} (Severity: {error['severity']})\n"
                md += f"  - Count: {error['count']} ({error['percentage']}%)\n"
                md += f"  - Message: {error['message']}\n"
                if error.get('samples'):
                    md += f"  - Sample invalid values: `{error['samples'][:3]}`\n"
                md += "\n"
        
        # Warnings
        if validation_summary.get('warning_count', 0) > 0:
            md += "### ⚠️ Warnings\n\n"
            for warning in validation_summary.get('warnings', []):
                md += f"- **{warning['dimension'].upper()}**\n"
                md += f"  - Type: {warning['warning_type']}\n"
                md += f"  - Message: {warning['message']}\n\n"
    
    md += f"""---

## 📈 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Transactions** | {eda_results['shape']['rows']:,} |
| **Total Columns** | {eda_results['shape']['columns']} |
| **Numeric Columns** | {eda_results['column_types']['numeric']} |
| **Text Columns** | {eda_results['column_types']['object']} |
| **Completeness** | {dq['scores']['completeness']:.2f}% |
| **Missing Cells** | {dq['dimensions']['completeness']['missing_cells']:,} |
| **Duplicate Records** | {dq['dimensions']['uniqueness']['duplicate_rows']:,} ({dq['dimensions']['uniqueness']['duplicate_percentage']}%) |

---

## 🔍 Missing Values Analysis

"""
    
    md += f"**Columns with Missing Data:** {len([c for c, s in dq['dimensions']['completeness']['column_scores'].items() if s < 100])}\n\n"
    
    # Show top columns with missing values
    missing_cols = {c: s for c, s in dq['dimensions']['completeness']['column_scores'].items() if s < 100}
    if missing_cols:
        sorted_missing = sorted(missing_cols.items(), key=lambda x: x[1])
        md += "| Column | Completeness % |\n|--------|----------------|\n"
        for col, score in sorted_missing[:10]:
            md += f"| `{col}` | {score:.2f}% |\n"
    else:
        md += "✅ **No missing values detected!**\n"
    
    md += "\n---\n\n## 🤖 AI Agent Analysis Results\n\n"
    
    summary = ai_analysis['summary']
    metrics = ai_analysis['metrics']
    
    md += f"""**Processing Summary:**
- Total Records Analyzed: {summary['total_records_processed']}
- Success Rate: {summary['success_rate']}
- Average Quality Score: {metrics['average_quality_score']:.3f}
- Total AI Insights: {metrics['total_insights_generated']}

"""
    
    if ai_analysis['detailed_results']:
        md += "### 📋 Sample Record Analysis\n\n"
        for i, result in enumerate(ai_analysis['detailed_results'][:3], 1):
            md += f"**Record {i}:** `{result['record_id']}`\n"
            md += f"- Quality Score: {result.get('quality_score', 'N/A')}\n"
            md += f"- Status: {result.get('status', 'unknown').upper()}\n"
            if result.get('ai_insights'):
                md += f"- Insights: {len(result['ai_insights'])}\n"
                for insight in result['ai_insights'][:2]:
                    md += f"  - _{insight}_\n"
            md += "\n"
    
    md += """---

## 💡 Computation Metadata

"""
    
    # Add metadata from data quality assessment
    metadata = dq.get('computation_metadata', {})
    md += f"- **No Simulation:** {metadata.get('no_simulation', 'N/A')}\n"
    md += f"- **All Scores Normalized (0-100):** {metadata.get('all_scores_normalized', 'N/A')}\n"
    md += f"- **LLM Insights Enabled:** {metadata.get('llm_enabled', 'N/A')}\n"
    md += f"- **Weights Applied:** {dq.get('weights_applied', 'N/A')}\n"
    
    md += "\n---\n\n*Report generated by Production-Grade Data Quality Assessment System*\n"
    md += "*All metrics computed from real data - No simulation or mocking*\n"
    
    return md


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return '⭐ A'
    elif score >= 80:
        return '✅ B'
    elif score >= 70:
        return '⚠️ C'
    elif score >= 60:
        return '❌ D'
    else:
        return '🚨 F'


@app.post("/analyze-transactions")
async def analyze_transactions(
    file: UploadFile = File(...),
    num_samples: int = 5
):
    """
    Upload a CSV file for comprehensive transaction analysis.
    
    Args:
        file: CSV file containing transaction data
        num_samples: Number of sample records to analyze with AI (default: 5, max: 20)
    
    Returns:
        JSON response with analysis results and markdown report
    """
    # Validate file type
    if not file.filename.endswith('.csv'): # type: ignore
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    # Limit sample size
    num_samples = min(max(num_samples, 1), 20)
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Extract dataset name from filename
        dataset_name = Path(file.filename or "dataset").stem
        
        # Perform EDA with error handling
        try:
            eda_results = perform_eda(df, dataset_name)
            
            # Check for critical validation failures
            validation_summary = eda_results.get('data_quality_dimensions', {}).get('validation_summary', {})
            if validation_summary.get('critical_count', 0) > 0:
                critical_failures = validation_summary.get('critical_failures', [])
                error_details = {
                    "error": "critical_validation_failures",
                    "message": "Dataset contains critical type safety violations",
                    "failures": critical_failures,
                    "recommendation": "Fix data type issues before reprocessing"
                }
                return JSONResponse(
                    status_code=422,
                    content=error_details
                )
        
        except ValueError as ve:
            raise HTTPException(
                status_code=422, 
                detail=f"Data validation error: {str(ve)}"
            )
        except Exception as eda_error:
            raise HTTPException(
                status_code=500,
                detail=f"EDA analysis failed: {str(eda_error)}"
            )
        
        # Create records for AI analysis
        try:
            records = create_records_from_csv(df, dataset_name, num_samples)
        except Exception as record_error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create records: {str(record_error)}"
            )
        
        # Process through AI agents with EDA context
        try:
            ai_analysis = await process_records_with_agents(records, eda_results)
        except Exception as agent_error:
            raise HTTPException(
                status_code=500,
                detail=f"AI agent processing failed: {str(agent_error)}"
            )
        
        # Generate markdown report
        try:
            markdown_report = generate_markdown_report(dataset_name, eda_results, ai_analysis)
        except Exception as report_error:
            # Non-critical - can still return data without markdown
            markdown_report = f"# Report Generation Failed\n\nError: {str(report_error)}"
        
        # Create comprehensive anomaly report
        anomaly_report = create_anomaly_report(eda_results)
        
        # Prepare JSON response
        response_data = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "eda_analysis": eda_results,
            "ai_analysis": ai_analysis,
            "anomaly_report": anomaly_report,  # NEW: Consolidated anomaly information
            "validation_summary": eda_results.get('data_quality_dimensions', {}).get('validation_summary', {}),
            "has_validation_errors": eda_results.get('data_quality_dimensions', {}).get('validation_summary', {}).get('has_issues', False),
            "markdown_report": markdown_report
        }
        
        # Sanitize response data to handle NaN/Infinity values
        sanitized_response = sanitize_for_json(response_data)
        
        return JSONResponse(content=sanitized_response)
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Failed to parse CSV file - check format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    """API health check and information."""
    return {
        "service": "Bank Transaction Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/analyze-transactions": "POST - Upload CSV for analysis",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
