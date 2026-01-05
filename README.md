# 📊 Transaction Analysis API - Frontend Integration Guide

**Version:** 1.0.0  
**Last Updated:** January 5, 2026  
**Base URL:** `http://localhost:8000` (or your deployed URL)

---

## 🎯 Quick Start

### Upload & Analyze Transactions

**Endpoint:** `POST /analyze-transactions`

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze-transactions?num_samples=5" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@transactions.csv"
```

**Response:** See complete JSON structure below ↓

---

## 📋 Response Structure Overview

```json
{
  "dataset_name": "SBI Transactions",
  "timestamp": "2026-01-05T01:48:50.398821",
  "eda_analysis": { /* Exploratory Data Analysis */ },
  "ai_analysis": { /* AI Agent Processing Results */ },
  "anomaly_report": { /* ⭐ NEW: Consolidated Anomaly Detection */ },
  "validation_summary": { /* Type Safety & Validation Errors */ },
  "has_validation_errors": true,
  "markdown_report": "..." /* Human-readable markdown */
}
```

---

## 🔑 Key Response Fields for Frontend

### 1️⃣ **Quick Health Check**

```typescript
interface QuickHealthCheck {
  dataset_name: string;
  timestamp: string; // ISO 8601
  has_validation_errors: boolean; // ⚠️ Use for alerts
}
```

**Example:**
```json
{
  "dataset_name": "SBI Transactions",
  "timestamp": "2026-01-05T01:48:50.398821",
  "has_validation_errors": true
}
```

**Frontend Use:**
- Show dataset name in header
- Display timestamp for audit trail
- Show warning banner if `has_validation_errors === true`

---

### 2️⃣ **Overall Quality Score** ⭐

**Path:** `eda_analysis.data_quality_dimensions.overall_score`

```typescript
interface QualityScoreSummary {
  overall_score: number; // 0-100
  quality_grade: string; // "A (Excellent)" | "B (Good)" | "C (Fair)" | "D (Poor)" | "F (Fail)"
  confidence_level: "high" | "medium" | "low";
}
```

**Example:**
```json
{
  "overall_score": 80.77,
  "quality_grade": "B (Good)",
  "confidence_level": "high"
}
```

**Frontend Display Ideas:**
```tsx
<ScoreCard>
  <CircularProgress value={80.77} color="green" />
  <Grade>B (Good)</Grade>
  <ConfidenceBadge level="high">High Confidence</ConfidenceBadge>
</ScoreCard>
```

---

### 3️⃣ **6 Dimension Scores** 📊

**Path:** `eda_analysis.data_quality_dimensions.scores`

```typescript
interface DimensionScores {
  completeness: number; // % of non-null values
  uniqueness: number;   // % of non-duplicate records
  validity: number;     // % passing format checks
  consistency: number;  // % following patterns
  accuracy: number;     // Outlier detection (ML)
  timeliness: number;   // Data freshness
}
```

**Example:**
```json
{
  "completeness": 90.49,
  "uniqueness": 99.94,
  "validity": 90.67,
  "consistency": 92.68,
  "accuracy": 90.15,
  "timeliness": 37.50  // ⚠️ FAILING
}
```

**Frontend Visualization:**

**Bar Chart:**
```tsx
<DimensionChart>
  <Bar name="Completeness" value={90.49} color="green" />
  <Bar name="Uniqueness" value={99.94} color="green" />
  <Bar name="Validity" value={90.67} color="green" />
  <Bar name="Consistency" value={92.68} color="green" />
  <Bar name="Accuracy" value={90.15} color="green" />
  <Bar name="Timeliness" value={37.50} color="red" />  {/* Red if < 60 */}
</DimensionChart>
```

**Color Coding:**
- 🟢 Green: 80-100
- 🟡 Yellow: 60-79
- 🔴 Red: 0-59

---

### 4️⃣ **⭐ Anomaly Report (NEW!)** 🚨

**Path:** `anomaly_report`

This is your **one-stop shop** for all anomaly information.

#### 4.1 Summary Dashboard

**Path:** `anomaly_report.summary`

```typescript
interface AnomalySummary {
  total_anomalies: number;
  critical_count: number;    // Fatal errors
  error_count: number;       // Type validation errors
  warning_count: number;     // Non-critical issues
  outlier_count: number;     // ML-detected anomalies
  has_critical_issues: boolean; // ⚠️ Use for STOP alert
  overall_anomaly_rate: number; // % of total data
}
```

**Example:**
```json
{
  "total_anomalies": 4162,
  "critical_count": 0,
  "error_count": 2,
  "warning_count": 2,
  "outlier_count": 4160,
  "has_critical_issues": false,
  "overall_anomaly_rate": 2.12
}
```

**Frontend Alert Component:**
```tsx
{anomaly_report.summary.has_critical_issues && (
  <CriticalAlert severity="error">
    🚨 Critical data issues detected! Processing stopped.
    See critical_failures for details.
  </CriticalAlert>
)}

{anomaly_report.summary.total_anomalies > 0 && (
  <WarningAlert severity="warning">
    ⚠️ {anomaly_report.summary.total_anomalies} anomalies detected
    ({anomaly_report.summary.overall_anomaly_rate}% of data)
  </WarningAlert>
)}
```

#### 4.2 ML-Detected Outliers

**Path:** `anomaly_report.ml_detected_outliers.by_column`

```typescript
interface ColumnOutlier {
  column: string;
  outlier_count: number;
  outlier_percentage: number;
  sample_values: any[]; // First 5 outlier values
  severity: "high" | "medium"; // high if >10%
}
```

**Example:**
```json
{
  "column": "transaction_amount_inr",
  "outlier_count": 351,
  "outlier_percentage": 10.02,
  "sample_values": [9777.06, 9870.3, -8633.98, -1291.06, -2950.17],
  "severity": "high"
}
```

**Frontend Table:**
```tsx
<OutlierTable>
  {anomaly_report.ml_detected_outliers.by_column
    .filter(o => o.severity === "high")
    .map(outlier => (
      <Row key={outlier.column}>
        <Cell>{outlier.column}</Cell>
        <Cell>
          <Badge color="red">{outlier.outlier_percentage}%</Badge>
        </Cell>
        <Cell>
          <Tooltip title={outlier.sample_values.join(', ')}>
            {outlier.outlier_count} outliers
          </Tooltip>
        </Cell>
      </Row>
    ))
  }
</OutlierTable>
```

#### 4.3 Type Validation Errors

**Path:** `anomaly_report.type_validation_errors.errors`

```typescript
interface ValidationError {
  column: string;
  error_type: string; // "invalid_date_format", "unparseable_datetime", etc.
  severity: "high" | "medium" | "low";
  count: number;
  percentage: number;
  sample_invalid_values: any[]; // First 3 bad values
  message: string;
  dimension: string; // Which quality dimension detected it
}
```

**Example:**
```json
{
  "column": "local_date",
  "error_type": "invalid_date_format",
  "severity": "high",
  "count": 1,
  "percentage": 0.03,
  "sample_invalid_values": ["Karan"],
  "message": "1 values could not be parsed as dates",
  "dimension": "validity"
}
```

**Frontend Display:**
```tsx
<ErrorList>
  {anomaly_report.type_validation_errors.errors.map((error, i) => (
    <ErrorCard key={i} severity={error.severity}>
      <ErrorHeader>
        <Icon severity={error.severity} />
        {error.column}
      </ErrorHeader>
      <ErrorDetails>
        <p><strong>Type:</strong> {error.error_type}</p>
        <p><strong>Count:</strong> {error.count} ({error.percentage}%)</p>
        <p><strong>Message:</strong> {error.message}</p>
        <CodeBlock>
          Invalid values: {error.sample_invalid_values.join(', ')}
        </CodeBlock>
      </ErrorDetails>
    </ErrorCard>
  ))}
</ErrorList>
```

#### 4.4 High-Risk Columns (Priority List)

**Path:** `anomaly_report.high_risk_columns`

```typescript
interface HighRiskColumn {
  column: string;
  risk_score: number; // Weighted score combining all anomaly types
  anomaly_types: string[]; // All detected anomalies
  severity: "critical" | "high" | "medium";
}
```

**Example:**
```json
{
  "column": "local_date",
  "risk_score": 75.0,
  "anomaly_types": [
    "invalid_date_format (high)",
    "unparseable_datetime (medium)"
  ],
  "severity": "high"
}
```

**Frontend Priority Table:**
```tsx
<HighRiskTable>
  <TableHeader>
    <th>Column</th>
    <th>Risk Score</th>
    <th>Issues</th>
    <th>Action</th>
  </TableHeader>
  {anomaly_report.high_risk_columns.slice(0, 5).map(col => (
    <Row key={col.column} severity={col.severity}>
      <Cell>
        <SeverityBadge severity={col.severity} />
        {col.column}
      </Cell>
      <Cell>
        <RiskMeter value={col.risk_score} max={100} />
      </Cell>
      <Cell>
        <ChipGroup>
          {col.anomaly_types.map(type => (
            <Chip size="small">{type}</Chip>
          ))}
        </ChipGroup>
      </Cell>
      <Cell>
        <Button size="small">Fix Now</Button>
      </Cell>
    </Row>
  ))}
</HighRiskTable>
```

---

### 5️⃣ **Dataset Statistics** 📈

**Path:** `eda_analysis.shape` + `eda_analysis.missing_values`

```typescript
interface DatasetStats {
  shape: {
    rows: number;
    columns: number;
  };
  missing_values: {
    total_columns_with_missing: number;
    columns: {
      [columnName: string]: {
        count: number;
        percentage: number;
      }
    }
  };
  duplicates: {
    count: number;
    percentage: number;
  };
}
```

**Example:**
```json
{
  "shape": {
    "rows": 3502,
    "columns": 56
  },
  "missing_values": {
    "total_columns_with_missing": 14,
    "columns": {
      "merchant_phone": {
        "count": 2105,
        "percentage": 60.11
      }
    }
  },
  "duplicates": {
    "count": 2,
    "percentage": 0.06
  }
}
```

**Frontend Cards:**
```tsx
<StatsGrid>
  <StatCard>
    <Label>Total Records</Label>
    <Value>{eda_analysis.shape.rows.toLocaleString()}</Value>
  </StatCard>
  
  <StatCard>
    <Label>Columns</Label>
    <Value>{eda_analysis.shape.columns}</Value>
  </StatCard>
  
  <StatCard>
    <Label>Missing Values</Label>
    <Value color="warning">
      {eda_analysis.missing_values.total_columns_with_missing} columns
    </Value>
  </StatCard>
  
  <StatCard>
    <Label>Duplicates</Label>
    <Value color={duplicates.count > 0 ? "warning" : "success"}>
      {duplicates.count} ({duplicates.percentage}%)
    </Value>
  </StatCard>
</StatsGrid>
```

---

### 6️⃣ **AI Agent Analysis** 🤖

**Path:** `ai_analysis.summary` + `ai_analysis.detailed_results`

```typescript
interface AIAnalysisSummary {
  summary: {
    total_records_processed: number;
    successful: number;
    failed: number;
    success_rate: string; // "100.00%"
  };
  metrics: {
    average_quality_score: number; // 0-1 scale
    total_insights_generated: number;
  };
  detailed_results: RecordAnalysis[];
}

interface RecordAnalysis {
  record_id: string;
  status: "completed" | "failed";
  quality_score: number; // 0-1 scale
  ai_insights: string[];
  predictions: {
    trend_forecast: "improving" | "stable" | "declining";
    confidence_score: number;
    predicted_quality_score_7d: number;
    predicted_quality_score_14d: number;
    predicted_quality_score_30d: number;
  };
  processing_time_ms: number;
  errors: any[];
}
```

**Example:**
```json
{
  "summary": {
    "total_records_processed": 5,
    "successful": 5,
    "failed": 0,
    "success_rate": "100.00%"
  },
  "metrics": {
    "average_quality_score": 0.795,
    "total_insights_generated": 5
  }
}
```

**Frontend Agent Dashboard:**
```tsx
<AgentDashboard>
  <ProgressBar 
    value={ai_analysis.summary.successful} 
    max={ai_analysis.summary.total_records_processed}
    label={`${ai_analysis.summary.success_rate} Success Rate`}
  />
  
  <MetricCards>
    <Card>
      <Label>Average Quality</Label>
      <Value>{(ai_analysis.metrics.average_quality_score * 100).toFixed(1)}%</Value>
    </Card>
    <Card>
      <Label>AI Insights</Label>
      <Value>{ai_analysis.metrics.total_insights_generated}</Value>
    </Card>
  </MetricCards>
  
  <RecordsList>
    {ai_analysis.detailed_results.map(record => (
      <RecordCard key={record.record_id}>
        <RecordHeader>
          <RecordID>{record.record_id}</RecordID>
          <StatusBadge status={record.status} />
        </RecordHeader>
        <QualityScore value={record.quality_score * 100} />
        <TrendForecast trend={record.predictions.trend_forecast} />
        <Insights items={record.ai_insights} />
      </RecordCard>
    ))}
  </RecordsList>
</AgentDashboard>
```

---

### 7️⃣ **LLM Insights** 🧠

**Path:** `eda_analysis.data_quality_dimensions.llm_insights`

```typescript
interface LLMInsights {
  summary: string; // Executive summary
  root_causes: string[]; // Identified root causes
  recommendations: string[]; // Action items
}
```

**Example:**
```json
{
  "summary": "The dataset exhibits strong performance in Completeness, Uniqueness, and Consistency, but **Accuracy (90.15)** and **Timeliness (37.50)** are critical risk areas...",
  "root_causes": [
    "[HIGH] Outdated Data Ingestion Pipelines (Timeliness)**",
    "[HIGH] Sensor/System Anomalies Causing Outliers (Accuracy)**"
  ],
  "recommendations": [
    "[HIGH] Implement Pipeline Monitoring & Automate Data Freshness Alerts**",
    "Metric Target**: Improve Timeliness to ≥85 within 30 days."
  ]
}
```

**Frontend Display:**
```tsx
<InsightsPanel>
  <Section>
    <SectionTitle>Executive Summary</SectionTitle>
    <Markdown>{llm_insights.summary}</Markdown>
  </Section>
  
  <Section>
    <SectionTitle>Root Causes</SectionTitle>
    <OrderedList>
      {llm_insights.root_causes.map((cause, i) => (
        <ListItem key={i}>
          <PriorityBadge>{extractPriority(cause)}</PriorityBadge>
          {cause}
        </ListItem>
      ))}
    </OrderedList>
  </Section>
  
  <Section>
    <SectionTitle>Recommended Actions</SectionTitle>
    <ActionList>
      {llm_insights.recommendations.map((rec, i) => (
        <ActionItem key={i}>
          <Checkbox />
          {rec}
        </ActionItem>
      ))}
    </ActionList>
  </Section>
</InsightsPanel>
```

---

## 🎨 Frontend Component Examples

### Complete Dashboard Layout

```tsx
import React from 'react';

interface AnalysisResponse {
  dataset_name: string;
  timestamp: string;
  eda_analysis: {
    data_quality_dimensions: {
      overall_score: number;
      quality_grade: string;
      scores: DimensionScores;
      llm_insights: LLMInsights;
    };
    shape: { rows: number; columns: number };
    missing_values: any;
    duplicates: any;
  };
  anomaly_report: AnomalyReport;
  ai_analysis: AIAnalysis;
  has_validation_errors: boolean;
}

function TransactionAnalysisDashboard({ data }: { data: AnalysisResponse }) {
  return (
    <DashboardLayout>
      {/* Header */}
      <Header>
        <Title>{data.dataset_name}</Title>
        <Timestamp>{new Date(data.timestamp).toLocaleString()}</Timestamp>
        {data.has_validation_errors && (
          <AlertBanner severity="warning">
            ⚠️ Validation errors detected
          </AlertBanner>
        )}
      </Header>

      {/* Overall Score */}
      <Section>
        <ScoreCard
          score={data.eda_analysis.data_quality_dimensions.overall_score}
          grade={data.eda_analysis.data_quality_dimensions.quality_grade}
        />
      </Section>

      {/* Anomaly Summary */}
      <Section>
        <AnomalySummaryCard data={data.anomaly_report.summary} />
      </Section>

      {/* 6 Dimensions */}
      <Section>
        <DimensionChart scores={data.eda_analysis.data_quality_dimensions.scores} />
      </Section>

      {/* High-Risk Columns */}
      <Section>
        <HighRiskTable columns={data.anomaly_report.high_risk_columns} />
      </Section>

      {/* LLM Insights */}
      <Section>
        <InsightsPanel data={data.eda_analysis.data_quality_dimensions.llm_insights} />
      </Section>

      {/* AI Agent Results */}
      <Section>
        <AgentResults data={data.ai_analysis} />
      </Section>
    </DashboardLayout>
  );
}
```

---

## 🚀 API Integration Examples

### React/TypeScript

```typescript
async function analyzeTransactions(file: File, numSamples: number = 5) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE_URL}/analyze-transactions?num_samples=${numSamples}`,
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data: AnalysisResponse = await response.json();
  return data;
}

// Usage
const handleFileUpload = async (file: File) => {
  try {
    setLoading(true);
    const result = await analyzeTransactions(file);
    
    // Check for critical issues
    if (result.anomaly_report.summary.has_critical_issues) {
      showCriticalAlert(result.anomaly_report.critical_failures);
      return;
    }
    
    // Display results
    setAnalysisData(result);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
};
```

### JavaScript/Fetch

```javascript
function uploadCSV(fileInput) {
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  fetch('http://localhost:8000/analyze-transactions?num_samples=5', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    console.log('Overall Score:', data.eda_analysis.data_quality_dimensions.overall_score);
    console.log('Total Anomalies:', data.anomaly_report.summary.total_anomalies);
    
    // Display high-risk columns
    data.anomaly_report.high_risk_columns.forEach(col => {
      console.log(`${col.column}: Risk Score ${col.risk_score}`);
    });
  })
  .catch(error => console.error('Error:', error));
}
```

---

## 📊 Key Metrics to Display

### Priority 1: Critical Alerts
```typescript
// Show FIRST in UI
if (data.anomaly_report.summary.has_critical_issues) {
  // STOP - show critical failures
}

if (data.has_validation_errors) {
  // WARNING - show validation errors
}
```

### Priority 2: Quality Score
```typescript
// Main dashboard metric
const overallScore = data.eda_analysis.data_quality_dimensions.overall_score;
const grade = data.eda_analysis.data_quality_dimensions.quality_grade;
```

### Priority 3: Anomaly Count
```typescript
// Show in summary card
const totalAnomalies = data.anomaly_report.summary.total_anomalies;
const anomalyRate = data.anomaly_report.summary.overall_anomaly_rate;
```

### Priority 4: High-Risk Columns
```typescript
// Show top 5 problematic columns
const topRisks = data.anomaly_report.high_risk_columns.slice(0, 5);
```

---

## 🎯 Error Handling

### HTTP Status Codes

| Code | Meaning | Response |
|------|---------|----------|
| 200 | Success | Full analysis JSON |
| 400 | Bad Request | Empty or invalid CSV |
| 422 | Validation Failed | Critical data type errors |
| 500 | Server Error | Processing failed |

### 422 Critical Failure Response

```json
{
  "error": "critical_validation_failures",
  "message": "Dataset contains critical type safety violations",
  "failures": [
    {
      "dimension": "timeliness",
      "column": "created_date",
      "error": "all_dates_invalid",
      "message": "All datetime values in created_date are invalid"
    }
  ],
  "recommendation": "Fix data type issues before reprocessing"
}
```

**Frontend Handling:**
```typescript
if (response.status === 422) {
  const errorData = await response.json();
  showCriticalError({
    title: errorData.message,
    failures: errorData.failures,
    recommendation: errorData.recommendation
  });
}
```

---

## 💡 Best Practices

### 1. **Progressive Loading**
```typescript
// Load in stages for large datasets
setLoading({ stage: 'uploading', progress: 0 });
// ... upload
setLoading({ stage: 'analyzing', progress: 50 });
// ... analysis
setLoading({ stage: 'complete', progress: 100 });
```

### 2. **Error Boundaries**
```tsx
<ErrorBoundary fallback={<ErrorPage />}>
  <AnalysisDashboard data={analysisData} />
</ErrorBoundary>
```

### 3. **Data Caching**
```typescript
// Cache results to avoid re-analysis
const cacheKey = `analysis_${file.name}_${file.lastModified}`;
localStorage.setItem(cacheKey, JSON.stringify(data));
```

### 4. **Responsive Design**
- Mobile: Show summary cards only
- Tablet: Add dimension chart
- Desktop: Full dashboard with tables

---

## 🔗 Additional Resources

- **Full API Spec:** `API_DOCUMENTATION.md`
- **Anomaly Report Schema:** `ANOMALY_REPORT_SCHEMA.md`
- **Validation System:** `VALIDATION_SYSTEM.md`

---

## 📞 Support

For issues or questions:
- GitHub Issues: [Shastra2026_Backend](https://github.com/kushvinth/Shastra2026_Backend)
- API Version: Check `/health` endpoint

**Health Check:**
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-05T01:48:50.398821"
}
```
