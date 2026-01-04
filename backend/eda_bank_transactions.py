"""
Exploratory Data Analysis (EDA) for Bank Transaction Datasets

This script performs comprehensive EDA on all transaction datasets and runs them
through the parallel agent system for intelligent analysis.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from src.record_processor import RecordOrchestrator, Record
import json

# Load environment variables
load_dotenv()


def load_datasets():
    """Load all bank transaction CSV files."""
    data_dir = Path("data")
    datasets = {}
    
    files = {
        "HDFC": "HDFC Transactions.csv",
        "SBI": "SBI Transactions.csv", 
        "IDBI": "IDBI Transactions.csv",
        "Multi": "Multi Bank Transactions.csv"
    }
    
    for bank, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n{'='*80}")
            print(f"📊 Loading {bank} dataset from: {filename}")
            print(f"{'='*80}")
            df = pd.read_csv(filepath)
            datasets[bank] = df
            print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        else:
            print(f"⚠️  Warning: {filename} not found")
    
    return datasets


def perform_eda(df, bank_name):
    """Perform comprehensive EDA on a dataset."""
    print(f"\n{'='*80}")
    print(f"🔍 EDA ANALYSIS FOR {bank_name}")
    print(f"{'='*80}\n")
    
    eda_results = {}
    
    # 1. Dataset Shape
    print("1️⃣  DATASET SHAPE")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    eda_results['shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}
    
    # 2. Column Names and Types
    print("\n2️⃣  COLUMN INFORMATION")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Object columns: {len(df.select_dtypes(include=['object']).columns)}")
    eda_results['column_types'] = df.dtypes.astype(str).to_dict()
    
    # 3. Missing Values Analysis
    print("\n3️⃣  MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_data = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_data) > 0:
        print(f"   Columns with missing values: {len(missing_data)}")
        print(f"\n   Top 10 columns with most missing values:")
        for col, row in missing_data.head(10).iterrows():
            print(f"   - {col}: {row['Missing_Count']:,} ({row['Percentage']}%)")
        eda_results['missing_values'] = {
            'total_columns_with_missing': len(missing_data),
            'top_missing': missing_data.head(10).to_dict('index')
        }
    else:
        print("   ✅ No missing values found!")
        eda_results['missing_values'] = {'total_columns_with_missing': 0}
    
    # 4. Numeric Columns Analysis
    print("\n4️⃣  NUMERIC COLUMNS STATISTICS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        print(f"   Analyzing {len(numeric_cols)} numeric columns...")
        numeric_stats = df[numeric_cols].describe()
        print(f"\n   Sample statistics for first numeric column:")
        if len(numeric_cols) > 0:
            first_col = numeric_cols[0]
            print(f"   {first_col}:")
            print(f"      Mean: {df[first_col].mean():.2f}")
            print(f"      Median: {df[first_col].median():.2f}")
            print(f"      Std: {df[first_col].std():.2f}")
            print(f"      Min: {df[first_col].min():.2f}")
            print(f"      Max: {df[first_col].max():.2f}")
        eda_results['numeric_stats'] = numeric_stats.to_dict()
    else:
        print("   No numeric columns found")
        eda_results['numeric_stats'] = {}
    
    # 5. Categorical Analysis
    print("\n5️⃣  CATEGORICAL COLUMNS ANALYSIS")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"   Analyzing {len(categorical_cols)} categorical columns...")
        print(f"\n   Unique value counts (top 5 columns):")
        for col in categorical_cols[:5]:
            unique_count = df[col].nunique()
            print(f"   - {col}: {unique_count:,} unique values")
        eda_results['categorical_info'] = {
            col: {'unique_count': int(df[col].nunique())} 
            for col in categorical_cols[:10]
        }
    else:
        print("   No categorical columns found")
        eda_results['categorical_info'] = {}
    
    # 6. Data Quality Metrics
    print("\n6️⃣  DATA QUALITY METRICS")
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"   Overall Completeness: {completeness:.2f}%")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df) * 100).round(2)
    print(f"   Duplicate Rows: {duplicate_count:,} ({duplicate_pct}%)")
    
    eda_results['data_quality'] = {
        'completeness': float(completeness),
        'duplicate_rows': int(duplicate_count),
        'duplicate_percentage': float(duplicate_pct)
    }
    
    # 7. Sample Records
    print("\n7️⃣  SAMPLE RECORDS")
    print(f"   First 3 rows preview:")
    print(df.head(3).to_string(max_cols=10, max_colwidth=20))
    
    return eda_results


def create_records_from_eda(datasets, eda_results):
    """Create Record objects from EDA results for agent processing."""
    records = []
    
    for bank_name, df in datasets.items():
        eda = eda_results[bank_name]
        
        # Sample 2 actual transactions from each bank
        sample_transactions = df.head(2).to_dict('records')
        
        for idx, transaction in enumerate(sample_transactions, 1):
            # Clean the transaction data - convert to simple types
            clean_transaction = {}
            for key, value in transaction.items():
                if pd.isna(value):
                    clean_transaction[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    clean_transaction[key] = float(value) if not pd.isna(value) else None
                else:
                    clean_transaction[key] = str(value) if value is not None else None
            
            record = Record(
                record_id=f"{bank_name}-TXN-{idx:03d}",
                data={
                    "bank": bank_name,
                    "dataset_rows": eda['shape']['rows'],
                    "dataset_columns": eda['shape']['columns'],
                    "completeness": eda['data_quality']['completeness'],
                    "duplicate_percentage": eda['data_quality']['duplicate_percentage'],
                    "missing_columns_count": eda['missing_values']['total_columns_with_missing'],
                    # Include sample transaction fields (limited to key fields)
                    **{k: v for i, (k, v) in enumerate(clean_transaction.items()) if i < 10}
                },
                metadata={
                    "source": "EDA",
                    "bank": bank_name,
                    "analysis_type": "transaction_quality"
                }
            )
            records.append(record)
    
    return records


async def main():
    """Main execution function."""
    print("\n" + "🏦 "*40)
    print("BANK TRANSACTION DATA QUALITY ANALYSIS WITH AI AGENTS")
    print("🏦 "*40)
    
    # Step 1: Load datasets
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASETS")
    print("="*80)
    datasets = load_datasets()
    
    if not datasets:
        print("❌ No datasets found! Please check the data directory.")
        return
    
    # Step 2: Perform EDA on each dataset
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    eda_results = {}
    for bank_name, df in datasets.items():
        eda_results[bank_name] = perform_eda(df, bank_name)
    
    # Step 3: Create records for agent processing
    print("\n" + "="*80)
    print("STEP 3: PREPARING RECORDS FOR AI AGENT ANALYSIS")
    print("="*80)
    
    records = create_records_from_eda(datasets, eda_results)
    print(f"\n✅ Created {len(records)} records from {len(datasets)} datasets")
    print(f"   Records per bank: {len(records) // len(datasets)}")
    
    # Step 4: Run through AI agents
    print("\n" + "="*80)
    print("STEP 4: RUNNING AI AGENT ANALYSIS")
    print("="*80)
    
    orchestrator = RecordOrchestrator()
    results = await orchestrator.process_batch(records)
    
    # Step 5: Generate final report
    print("\n" + "="*80)
    print("📊 FINAL EDA & AI ANALYSIS REPORT")
    print("="*80)
    
    summary = orchestrator.get_summary()
    
    print(f"\n🏦 DATASETS ANALYZED: {len(datasets)}")
    for bank_name, df in datasets.items():
        eda = eda_results[bank_name]
        print(f"\n{bank_name} Bank:")
        print(f"  📝 Total Transactions: {eda['shape']['rows']:,}")
        print(f"  📊 Data Completeness: {eda['data_quality']['completeness']:.2f}%")
        print(f"  ⚠️  Missing Value Columns: {eda['missing_values']['total_columns_with_missing']}")
        print(f"  🔄 Duplicate Records: {eda['data_quality']['duplicate_rows']:,} ({eda['data_quality']['duplicate_percentage']}%)")
    
    print(f"\n🤖 AI AGENT ANALYSIS RESULTS:")
    print(f"  Total Records Processed: {summary['total_records']}")
    print(f"  ✅ Completed: {summary['completed']}")
    print(f"  ❌ Failed: {summary['failed']}")
    print(f"  ⏱️  Avg Processing Time: {summary['avg_processing_time_ms']:.0f}ms per record")
    
    print("\n" + "-"*80)
    print("AI Quality Insights by Bank:")
    print("-"*80)
    
    for result in summary['results']:
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        quality = f"{result['quality_score']:.3f}" if result['quality_score'] else "N/A"
        print(f"{status_icon} {result['record_id']}: Quality Score={quality}, AI Insights={result['insights_count']}")
    
    # Save detailed report
    report_file = "eda_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'eda_results': {k: {
                'shape': v['shape'],
                'data_quality': v['data_quality'],
                'missing_values': v['missing_values']
            } for k, v in eda_results.items()},
            'ai_analysis': summary
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    print("\n✨ Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
