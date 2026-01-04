// API Response Types based on FRONTEND_API_DOCUMENTATION.md

export interface AnalysisResponse {
    dataset_name: string;
    timestamp: string;
    eda_analysis: EDAAnalysis;
    ai_analysis: AIAnalysis;
    anomaly_report: AnomalyReport;
    validation_summary: ValidationSummary;
    has_validation_errors: boolean;
    markdown_report: string;
}

export interface EDAAnalysis {
    shape: {
        rows: number;
        columns: number;
    };
    missing_values: {
        total_columns_with_missing: number;
        columns: Record<string, { count: number; percentage: number }>;
    };
    duplicates: {
        count: number;
        percentage: number;
    };
    data_quality_dimensions: DataQualityDimensions;
}

export interface DataQualityDimensions {
    overall_score: number;
    quality_grade: string;
    confidence_level: "high" | "medium" | "low";
    scores: DimensionScores;
    llm_insights: LLMInsights;
}

export interface DimensionScores {
    completeness: number;
    uniqueness: number;
    validity: number;
    consistency: number;
    accuracy: number;
    timeliness: number;
}

export interface LLMInsights {
    summary: string;
    root_causes: string[];
    recommendations: string[];
}

export interface AIAnalysis {
    summary: {
        total_records_processed: number;
        successful: number;
        failed: number;
        success_rate: string;
    };
    metrics: {
        average_quality_score: number;
        total_insights_generated: number;
    };
    detailed_results: RecordAnalysis[];
}

export interface RecordAnalysis {
    record_id: string;
    status: "completed" | "failed";
    quality_score: number;
    ai_insights: string[];
    predictions: {
        trend_forecast: "improving" | "stable" | "declining";
        confidence_score: number;
        predicted_quality_score_7d: number;
        predicted_quality_score_14d: number;
        predicted_quality_score_30d: number;
    };
    processing_time_ms: number;
    errors: string[];
}

export interface AnomalyReport {
    summary: AnomalySummary;
    ml_detected_outliers: {
        total_outliers: number;
        by_column: ColumnOutlier[];
    };
    type_validation_errors: {
        total_errors: number;
        errors: ValidationError[];
    };
    high_risk_columns: HighRiskColumn[];
    critical_failures: CriticalFailure[];
}

export interface AnomalySummary {
    total_anomalies: number;
    critical_count: number;
    error_count: number;
    warning_count: number;
    outlier_count: number;
    has_critical_issues: boolean;
    overall_anomaly_rate: number;
}

export interface ColumnOutlier {
    column: string;
    outlier_count: number;
    outlier_percentage: number;
    sample_values: (string | number)[];
    severity: "high" | "medium";
}

export interface ValidationError {
    column: string;
    error_type: string;
    severity: "high" | "medium" | "low";
    count: number;
    percentage: number;
    sample_invalid_values: (string | number)[];
    message: string;
    dimension: string;
}

export interface HighRiskColumn {
    column: string;
    risk_score: number;
    anomaly_types: string[];
    severity: "critical" | "high" | "medium";
}

export interface CriticalFailure {
    dimension: string;
    column: string;
    error: string;
    message: string;
}

export interface ValidationSummary {
    total_validations: number;
    passed: number;
    failed: number;
}

// API Error Response
export interface APIError {
    error: string;
    message: string;
    failures?: CriticalFailure[];
    recommendation?: string;
}
