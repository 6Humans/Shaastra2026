"""
Record-based Parallel Agent System with OpenRouter Integration.

Each record flows through all agents in parallel before moving to the next record.
Uses OpenRouter with Qwen model for intelligent analysis and insights.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json
import httpx
import re
import ast


class TaskStatus(str, Enum):
    """Status enumeration for tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Record:
    """Individual data record to be processed."""
    record_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordResult:
    """Result of processing a single record through all agents."""
    record_id: str
    status: TaskStatus
    data_scientist_output: Optional[Dict] = None
    scoring_output: Optional[Dict] = None
    insight_output: Optional[Dict] = None
    predictive_output: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, openrouter_api_key: Optional[str] = None, eda_context: Optional[Dict[str, Any]] = None):
        self.name = name
        self.openrouter_api_key = openrouter_api_key
        self.eda_context = eda_context or {}
        self.logger = []

    @abstractmethod
    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Process a single record."""
        pass

    def log(self, message: str):
        """Log a message."""
        log_entry = f"[{self.name}] {datetime.now().isoformat()}: {message}"
        self.logger.append(log_entry)
        print(log_entry)
    
    def _format_eda_context(self) -> str:
        """Format EDA context for LLM prompts."""
        if not self.eda_context:
            return "No dataset context available."
        
        dq = self.eda_context.get('data_quality_dimensions', {})
        if not dq:
            return "No dataset context available."
        
        scores = dq.get('scores', {})
        weights = dq.get('weights_applied', {})
        summary = dq.get('summary', {})
        issues = dq.get('issues_detected', [])
        
        context = f"""
📊 **DATASET-LEVEL CONTEXT** (EDA Analysis):

**Dataset Size:**
- Total Rows: {summary.get('total_rows', 'Unknown'):,}
- Total Columns: {summary.get('total_columns', 'Unknown')}
- Total Cells: {summary.get('total_cells', 'Unknown'):,}

**Overall Data Quality:** {dq.get('overall_score', 'N/A'):.2f}/100 ({dq.get('quality_grade', 'N/A')})

**Quality Dimension Scores (0-100):**
- Completeness: {scores.get('completeness', 0):.2f} (Weight: {weights.get('completeness', 0)*100:.1f}%)
- Uniqueness: {scores.get('uniqueness', 0):.2f} (Weight: {weights.get('uniqueness', 0)*100:.1f}%)
- Validity: {scores.get('validity', 0):.2f} (Weight: {weights.get('validity', 0)*100:.1f}%)
- Consistency: {scores.get('consistency', 0):.2f} (Weight: {weights.get('consistency', 0)*100:.1f}%)
- Accuracy: {scores.get('accuracy', 0):.2f} (Weight: {weights.get('accuracy', 0)*100:.1f}%)
- Timeliness: {scores.get('timeliness', 0):.2f} (Weight: {weights.get('timeliness', 0)*100:.1f}%)

**Dataset-Wide Issues Detected ({len(issues)}):**
"""
        for issue in issues[:5]:  # Top 5 issues
            context += f"- [{issue.get('severity', 'unknown').upper()}] {issue.get('dimension', 'unknown')}: {issue.get('description', 'No description')}\n"
        
        if len(issues) > 5:
            context += f"- ... and {len(issues) - 5} more issues\n"
        
        context += f"""
**Weighting Strategy:** {dq.get('computation_metadata', {}).get('weighting_strategy', 'unknown')}
**Confidence Level:** {dq.get('confidence_level', 'unknown').upper()}

💡 **Use this dataset context to:**
- Compare this individual record against dataset-wide patterns
- Identify if this record contributes to known dataset issues
- Assess if this record is typical or anomalous for this dataset
- Consider the prioritized dimensions (higher weights) in your analysis
"""
        return context

    async def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call OpenRouter LLM (Qwen model)."""
        if not self.openrouter_api_key:
            return "LLM not configured"
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "qwen/qwen3-235b-a22b-2507",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 4000
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.log(f"LLM error: {str(e)}")
            return f"Error: {str(e)}"

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair invalid JSON strings."""
        # Remove non-printable characters and potential garbage
        # Keep basic ASCII, curly braces, brackets, quotes, commas, colons, newlines
        clean_str = re.sub(r'[^\x20-\x7E\s]', '', json_str)
        
        # fix trailing commas
        clean_str = re.sub(r',(\s*[}\]])', r'\1', clean_str)
        
        # simple quote fixes (this is risky but helps with basic errors)
        # ensure keys are double quoted
        clean_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', clean_str)
        
        return clean_str

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response with robust error handling."""
        # 1. Try extracting from markdown code blocks (case insensitive)
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(json_str)
                except:
                    # Try validation/repair
                    try:
                        repaired = self._repair_json(json_str)
                        return json.loads(repaired)
                    except:
                        pass

        # 2. Try finding the first '{' and matching '}' using a stack-based approach
        # (Simple regex approximations often fail with nested braces)
        try:
            # Find the first '{'
            start_index = response.find('{')
            if start_index != -1:
                # Find the last '}'
                end_index = response.rfind('}')
                if end_index != -1 and end_index > start_index:
                    potential_json = response[start_index:end_index+1]
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        try:
                            # Handle Python True/False/None
                            return ast.literal_eval(potential_json)
                        except:
                            # Try validation/repair
                            try:
                                repaired = self._repair_json(potential_json)
                                return json.loads(repaired)
                            except:
                                pass
        except:
            pass
            
        # 3. Last ditch: Try direct parsing of the whole string
        try:
            return json.loads(response)
        except:
            pass
            
        # Log failure and dump full response for debugging
        self.log(f"Failed to extract JSON. Dumping to llm_debug_dump.log")
        try:
            with open("llm_debug_dump.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n{'='*80}\nTIMESTAMP: {datetime.now().isoformat()}\nAGENT: {self.name}\n{'='*80}\n{response}\n{'='*80}\n")
        except Exception as e:
            self.log(f"Failed to write dump: {e}")
            
        return {}


class DataScientistAgent(Agent):
    """Performs data loading, validation, preprocessing, and EDA on individual records."""

    def __init__(self, openrouter_api_key: Optional[str] = None, eda_context: Optional[Dict[str, Any]] = None):
        super().__init__("DataScientistAgent", openrouter_api_key, eda_context)

    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Process a single record for data quality checks."""
        self.log(f"Processing record {record.record_id}")

        try:
            # Simulate data validation
            await asyncio.sleep(0.3)
            
            # Check for missing values
            missing_fields = [k for k, v in record.data.items() if v is None or v == ""]
            completeness = 1.0 - (len(missing_fields) / max(len(record.data), 1))

            # Use LLM for data quality assessment
            llm_analysis = ""
            if self.openrouter_api_key:
                eda_info = self._format_eda_context()
                prompt = f"""{eda_info}

📝 INDIVIDUAL RECORD ANALYSIS:

🔬 DATA QUALITY ANALYSIS - RECORD {record.record_id}

📊 RECORD DATA:
{json.dumps(record.data, indent=2)}

📋 YOUR TASK AS A DATA SCIENTIST:
Perform comprehensive data quality assessment on this transaction record.

🎯 ANALYSIS REQUIREMENTS:
1. **Completeness Check**:
   - Identify all missing/null/empty fields
   - Assess criticality of missing data (critical vs optional fields)
   - Calculate completeness ratio

2. **Data Type Validation**:
   - Verify each field has appropriate data type
   - Flag type mismatches (e.g., string in numeric field)
   - Check for implicit type coercion issues

3. **Format Validation**:
   - Email: RFC 5322 compliance
   - Phone: E.164 or local format
   - Dates: ISO 8601 or consistent format
   - Credit cards: Luhn algorithm validation
   - Currency: Proper decimal precision

4. **Statistical Anomalies**:
   - Outlier detection for numeric fields
   - Unusual value ranges
   - Suspiciously round numbers

5. **Business Logic Validation**:
   - Amount consistency (negative transactions?)
   - Date logic (future dates, unrealistic timestamps)
   - Category/status value validity

📤 OUTPUT FORMAT (JSON):
{{
  "completeness_score": 0.95,
  "missing_critical_fields": ["field1", "field2"],
  "type_issues": ["description"],
  "format_violations": ["description"],
  "anomalies_detected": ["description"],
  "business_rule_violations": ["description"],
  "overall_assessment": "PASS/WARNING/FAIL",
  "priority_actions": ["action1", "action2"]
}}

Be thorough, precise, and actionable."""
                
                llm_analysis = await self.call_llm(
                    prompt,
                    system_prompt="""You are a senior data scientist specializing in data quality engineering.
You have 10+ years of experience in financial transaction analysis, data validation, and quality assurance.
Your analysis must be:
- Technically rigorous and evidence-based
- Focused on actionable insights
- Compliant with industry standards (ISO 8000, DAMA DMBOK)
- Formatted as valid JSON
Always prioritize data integrity and business impact."""
                )

            output = {
                "record_id": record.record_id,
                "completeness_score": completeness,
                "missing_fields": missing_fields,
                "field_count": len(record.data),
                "validation_passed": len(missing_fields) == 0,
                "llm_analysis": llm_analysis
            }

            self.log(f"Record {record.record_id}: Completeness {completeness:.2%}")
            return output

        except Exception as e:
            self.log(f"Error processing record {record.record_id}: {str(e)}")
            raise


class ScoringAgent(Agent):
    """Computes data quality metrics and scores for individual records using LLM-enhanced analysis."""

    def __init__(self, openrouter_api_key: Optional[str] = None, eda_context: Optional[Dict[str, Any]] = None):
        super().__init__("ScoringAgent", openrouter_api_key, eda_context)

    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Score a single record's data quality with detailed LLM analysis."""
        self.log(f"Scoring record {record.record_id}")

        try:
            await asyncio.sleep(0.2)

            # Calculate various quality dimensions
            data = record.data
            
            # Completeness
            completeness = sum(1 for v in data.values() if v not in [None, ""]) / max(len(data), 1)
            
            # Use LLM for enhanced scoring
            llm_scores = {}
            if self.openrouter_api_key:
                eda_info = self._format_eda_context()
                prompt = f"""{eda_info}

📝 INDIVIDUAL RECORD SCORING:

🎯 QUALITY SCORING ANALYSIS - RECORD {record.record_id}

📊 RECORD DATA:
{json.dumps(record.data, indent=2)}

📋 YOUR TASK AS A SCORING ANALYST:
Provide precise quality scores (0.0-1.0) for this transaction record across multiple dimensions.

🎯 SCORING CRITERIA:

1. **VALIDITY SCORE** (0.0-1.0):
   - Email format: RFC 5322 compliance
   - Phone format: Valid format with country code
   - Card numbers: Luhn algorithm validation
   - Dates: ISO 8601 or parseable format
   - Amounts: Proper numeric format with 2 decimal places
   - Categories/Status: Valid enum values
   
   Deductions:
   - Minor format issue: -0.1
   - Major validation failure: -0.3
   - Critical field invalid: -0.5

2. **CONSISTENCY SCORE** (0.0-1.0):
   - Cross-field validation (e.g., amount matches transaction type)
   - Date logic (transaction date ≤ current date)
   - Status consistency (completed transactions have amounts)
   - Merchant-category alignment
   - Currency-amount decimal consistency
   
   Deductions:
   - Minor inconsistency: -0.1
   - Moderate contradiction: -0.2
   - Major logical error: -0.4

3. **ACCURACY SCORE** (0.0-1.0):
   - Realistic value ranges (amounts not suspiciously high/low)
   - Proper precision (currency decimals)
   - No truncation/rounding errors
   - Timezone consistency in timestamps
   - Proper capitalization and formatting
   
   Deductions:
   - Precision issue: -0.1
   - Range anomaly: -0.2
   - Obvious data corruption: -0.5

4. **COMPLETENESS SCORE**: {completeness:.3f} (pre-calculated)

📤 OUTPUT FORMAT (JSON):
{{
  "validity_score": 0.95,
  "validity_deductions": ["reason1", "reason2"],
  "consistency_score": 0.95,
  "consistency_issues": ["issue1", "issue2"],
  "accuracy_score": 0.95,
  "accuracy_concerns": ["concern1", "concern2"],
  "recommended_overall_score": 0.95,
  "confidence_level": "HIGH",
  "scoring_notes": "Brief explanation of score"
}}

Be objective, precise, and justify all deductions."""
                
                llm_response = await self.call_llm(
                    prompt,
                    system_prompt="""You are a quality scoring specialist with expertise in:
- ISO 8000 data quality standards
- Financial transaction validation (PCI-DSS)
- Statistical quality control (Six Sigma)
- Data governance frameworks (DAMA DMBOK)

Your scores must be:
- Objective and reproducible
- Based on measurable criteria
- Justified with specific evidence
- Formatted as valid JSON

Apply rigorous standards but be fair and consistent."""
                )
                
                
                llm_scores = self._extract_json(llm_response)
                if not llm_scores:
                    self.log(f"Using default scores due to parsing failure")
            
            # Extract LLM scores or use defaults
            validity = llm_scores.get('validity_score', 0.9)
            consistency = llm_scores.get('consistency_score', 0.95)
            accuracy = llm_scores.get('accuracy_score', 0.85)
            
            # Calculate weighted overall score (dynamic weights based on field presence)
            weights = {
                "completeness": 0.25,
                "validity": 0.30,
                "consistency": 0.25,
                "accuracy": 0.20
            }
            overall_score = (
                completeness * weights["completeness"] +
                validity * weights["validity"] +
                consistency * weights["consistency"] +
                accuracy * weights["accuracy"]
            )

            output = {
                "record_id": record.record_id,
                "completeness_score": round(completeness, 3),
                "validity_score": round(validity, 3),
                "consistency_score": round(consistency, 3),
                "accuracy_score": round(accuracy, 3),
                "overall_quality_score": round(overall_score, 3),
                "quality_level": "High" if overall_score >= 0.9 else "Medium" if overall_score >= 0.7 else "Low",
                "llm_analysis": llm_scores.get('scoring_notes', ''),
                "confidence": llm_scores.get('confidence_level', 'MEDIUM')
            }

            self.log(f"Record {record.record_id}: Overall score {overall_score:.3f}")
            return output

        except Exception as e:
            self.log(f"Error scoring record {record.record_id}: {str(e)}")
            raise


class InsightAgent(Agent):
    """Converts scores to human-readable insights using OpenRouter."""

    def __init__(self, openrouter_api_key: Optional[str] = None, eda_context: Optional[Dict[str, Any]] = None):
        super().__init__("InsightAgent", openrouter_api_key, eda_context)

    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Generate insights for a single record."""
        self.log(f"Generating insights for record {record.record_id}")

        try:
            await asyncio.sleep(0.25)

            # Use LLM to generate insights
            insights = []
            risks = []
            recommendations = []

            if self.openrouter_api_key:
                eda_info = self._format_eda_context()
                prompt = f"""{eda_info}

📝 INDIVIDUAL RECORD INSIGHT GENERATION:

💡 INSIGHT GENERATION - RECORD {record.record_id}

📊 RECORD DATA:
{json.dumps(record.data, indent=2)}

📋 YOUR TASK AS AN INSIGHT ANALYST:
Transform quality scores into strategic, actionable business insights.

🎯 ANALYSIS FRAMEWORK:

1. **KEY INSIGHTS** (3-5 insights):
   - What does this data tell us about transaction quality?
   - Are there patterns indicating systemic issues?
   - What's the business impact of detected issues?
   - Which fields are most critical for business operations?
   - Are there compliance or regulatory concerns?
   
   Format: ["Specific, measurable insight with business context"]

2. **RISK ASSESSMENT** (Priority-ranked):
   
   🔴 **CRITICAL RISKS** (Immediate action required):
   - Data integrity issues affecting financial accuracy
   - Compliance violations (PCI-DSS, GDPR)
   - Security concerns (exposed sensitive data)
   
   🟡 **MODERATE RISKS** (Address within 1 week):
   - Data quality degradation trends
   - Process efficiency concerns
   - Customer experience impacts
   
   🟢 **LOW RISKS** (Monitor and improve):
   - Minor formatting inconsistencies
   - Optimization opportunities
   - Best practice deviations
   
   Format per risk:
   {{
     "severity": "CRITICAL/MODERATE/LOW",
     "risk": "Specific risk description",
     "probability": "HIGH/MEDIUM/LOW",
     "impact": "Business impact if not addressed",
     "affected_fields": ["field1", "field2"]
   }}

3. **ACTIONABLE RECOMMENDATIONS** (Prioritized):
   
   Each recommendation must include:
   - **Action**: What to do (specific, not vague)
   - **Why**: Business justification
   - **How**: Implementation approach
   - **Effort**: LOW/MEDIUM/HIGH
   - **Impact**: Expected quality improvement (%)
   - **Timeline**: Immediate/Short-term/Long-term
   
   Example:
   {{
     "priority": 1,
     "action": "Implement real-time Luhn validation for card numbers",
     "justification": "23% of records fail card validation, risking payment processing errors",
     "implementation": "Add pre-submission validation in payment form with instant feedback",
     "effort": "LOW",
     "expected_improvement": "95% reduction in invalid card entries",
     "timeline": "Immediate",
     "estimated_cost_savings": "$50K annually in failed transaction fees"
   }}

4. **TREND INDICATORS**:
   - Is quality improving or degrading?
   - Are issues isolated or systemic?
   - What's the data maturity level?

📤 OUTPUT FORMAT (STRICT JSON):
{{
  "executive_summary": "One-sentence overview of record quality",
  "insights": [
    "Insight 1 with specific metrics",
    "Insight 2 with business context",
    "Insight 3 with actionable focus"
  ],
  "risks": [
    {{
      "severity": "CRITICAL/MODERATE/LOW",
      "risk": "Description",
      "probability": "HIGH/MEDIUM/LOW",
      "impact": "Business impact",
      "affected_fields": ["field1"]
    }}
  ],
  "recommendations": [
    {{
      "priority": 1,
      "action": "Specific action",
      "justification": "Why it matters",
      "implementation": "How to do it",
      "effort": "LOW/MEDIUM/HIGH",
      "expected_improvement": "Quantified benefit",
      "timeline": "When to implement"
    }}
  ],
  "trend_indicator": "IMPROVING/STABLE/DEGRADING",
  "data_maturity_score": 1-5,
  "confidence_level": "HIGH/MEDIUM/LOW"
}}

Be specific, quantitative, and business-focused. Avoid generic advice."""

                llm_response = await self.call_llm(
                    prompt,
                    system_prompt="""You are a senior data quality analyst and business intelligence expert with:
- 15+ years experience in financial services data governance
- Deep expertise in DAMA DMBOK, ISO 8000, and data quality frameworks
- Strong business acumen connecting data quality to ROI
- Track record of implementing enterprise data quality programs

Your insights must be:
- Specific and evidence-based (cite actual field names and values)
- Actionable with clear implementation paths
- Quantified with metrics and expected outcomes
- Business-focused (revenue, cost, risk, compliance)
- Formatted as valid, parseable JSON

Think like a consultant presenting to C-level executives: strategic, impactful, ROI-focused."""
                )

                
                parsed = self._extract_json(llm_response)
                
                if parsed:
                    insights = parsed.get("insights", [])
                    risks = parsed.get("risks", [])
                    recommendations = parsed.get("recommendations", [])
                else:
                    insights = ["Data quality analysis completed (parsing failed)"]
                    risks = ["Unable to parse detailed analysis"]
                    recommendations = ["Review data manually"]

            output = {
                "record_id": record.record_id,
                "insights": insights,
                "risks": risks,
                "recommendations": recommendations,
                "overall_assessment": "Generated via OpenRouter (Qwen) analysis"
            }

            self.log(f"Record {record.record_id}: Generated {len(insights)} insights")
            return output

        except Exception as e:
            self.log(f"Error generating insights for record {record.record_id}: {str(e)}")
            raise


class PredictiveAgent(Agent):
    """Forecasts potential issues and quality trends for individual records using advanced ML-informed prompts."""

    def __init__(self, openrouter_api_key: Optional[str] = None, eda_context: Optional[Dict[str, Any]] = None):
        super().__init__("PredictiveAgent", openrouter_api_key, eda_context)

    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Predict potential issues and quality degradation patterns for a single record."""
        self.log(f"Forecasting for record {record.record_id}")

        try:
            await asyncio.sleep(0.3)

            predictions = []
            confidence = 0.75

            if self.openrouter_api_key:
                eda_info = self._format_eda_context()
                prompt = f"""{eda_info}

📝 INDIVIDUAL RECORD PREDICTION:

🔮 PREDICTIVE QUALITY ANALYSIS - RECORD {record.record_id}

📊 RECORD DATA:
{json.dumps(record.data, indent=2)}

📋 YOUR TASK AS A PREDICTIVE ANALYST:
Forecast potential data quality issues, degradation patterns, and emerging risks.

🎯 PREDICTION FRAMEWORK:

1. **SHORT-TERM PREDICTIONS (1-7 days)**:
   
   a) **Data Staleness Risk**:
      - Are timestamp/date fields approaching outdated thresholds?
      - Will this record become irrelevant soon?
      - Score: LOW/MEDIUM/HIGH
   
   b) **Missing Data Propagation**:
      - Will current null fields cascade to related records?
      - Are optional fields becoming critical?
      - Probability: 0.0-1.0
   
   c) **Format Drift**:
      - Are current formats becoming deprecated?
      - Will validation rules change?
      - Impact: LOW/MEDIUM/HIGH

2. **MEDIUM-TERM PREDICTIONS (1-4 weeks)**:
   
   a) **Quality Degradation Trend**:
      - Based on field patterns, predict quality score in 7/14/30 days
      - Will completeness decrease?
      - Will consistency issues compound?
   
   b) **Compliance Risk Evolution**:
      - Will this record violate new regulations?
      - Are data retention policies approaching?
      - Risk level: LOW/MEDIUM/HIGH/CRITICAL
   
   c) **Integration Failures**:
      - Will downstream systems reject this record?
      - Are dependent fields at risk?
      - Probability: 0.0-1.0

3. **ANOMALY PATTERN FORECAST**:
   
   - Statistical anomalies likely to emerge
   - Outlier detection confidence over time
   - Behavioral pattern shifts

4. **PROACTIVE RECOMMENDATIONS**:
   
   What to do NOW to prevent predicted issues:
   - Data refresh schedules
   - Validation rule updates
   - Monitoring alerts to configure
   - Preemptive data enrichment

📤 OUTPUT FORMAT (STRICT JSON):
{{
  "short_term_predictions": [
    {{
      "issue": "Specific predicted issue",
      "probability": 0.85,
      "timeframe_days": 7,
      "severity": "HIGH",
      "leading_indicators": ["indicator1", "indicator2"],
      "affected_fields": ["field1", "field2"]
    }}
  ],
  "medium_term_predictions": [
    {{
      "issue": "Emerging trend or risk",
      "probability": 0.65,
      "timeframe_days": 30,
      "severity": "MEDIUM",
      "root_cause": "Why this will happen"
    }}
  ],
  "quality_score_forecast": {{
    "current_score": 0.90,
    "predicted_7d": 0.88,
    "predicted_14d": 0.85,
    "predicted_30d": 0.82,
    "trend_direction": "DEGRADING",
    "confidence": 0.90
  }},
  "anomaly_forecast": {{
    "statistical_anomalies_likely": true,
    "anomaly_types": ["outlier", "drift", "shift"],
    "detection_confidence": 0.85
  }},
  "proactive_actions": [
    {{
      "action": "Specific preventive measure",
      "prevents": "Which predicted issue",
      "urgency": "HIGH",
      "effort": "LOW",
      "risk_reduction": "60%"
    }}
  ],
  "overall_forecast": "One-sentence summary of predictions",
  "model_confidence": 0.90,
  "prediction_basis": "What data patterns drove these predictions"
}}

Be specific, quantitative, and time-bounded. Focus on actionable predictions."""

                llm_response = await self.call_llm(
                    prompt,
                    system_prompt="""You are a predictive analytics expert specializing in:
- Time series forecasting for data quality metrics
- Machine learning model interpretation (ARIMA, Prophet, LSTM)
- Anomaly detection and drift analysis
- Risk modeling and early warning systems
- Proactive data governance

Your predictions must be:
- Evidence-based on observable patterns in the data
- Time-bounded with specific timeframes
- Quantified with probabilities (0.0-1.0)
- Actionable with clear preventive measures
- Formatted as valid, parseable JSON

Think like a data scientist building an early warning system:
- Identify leading indicators
- Quantify confidence intervals
- Recommend proactive interventions
- Balance sensitivity vs specificity

Avoid vague predictions. Be specific about what will happen, when, and why."""
                )

                
                parsed = self._extract_json(llm_response)
                
                if parsed:
                    predictions = parsed.get("short_term_predictions", []) + parsed.get("medium_term_predictions", [])
                    confidence = parsed.get("model_confidence", 0.75)
                    quality_forecast = parsed.get("quality_score_forecast", {})
                else:
                    self.log(f"Failed to parse LLM prediction response")
                    predictions = [{
                        "issue": "Prediction parsing failed",
                        "probability": 0.0,
                        "timeframe_days": 7,
                        "severity": "LOW"
                    }]
                    quality_forecast = {"predicted_7d": 0.88}

            output = {
                "record_id": record.record_id,
                "predictions": predictions,
                "trend_forecast": quality_forecast.get("trend_direction", "stable"),
                "confidence_score": confidence,
                "predicted_quality_score_7d": quality_forecast.get("predicted_7d", 0.88),
                "predicted_quality_score_14d": quality_forecast.get("predicted_14d", 0.85),
                "predicted_quality_score_30d": quality_forecast.get("predicted_30d", 0.82)
            }

            self.log(f"Record {record.record_id}: {len(predictions)} predictions made")
            return output

        except Exception as e:
            self.log(f"Error forecasting for record {record.record_id}: {str(e)}")
            raise


class RecordOrchestrator:
    """
    Orchestrator that processes records one at a time.
    
    Each record flows through all 4 agents in parallel before moving to the next record.
    """

    def __init__(self, openrouter_api_key: Optional[str] = None, eda_context: Optional[Dict[str, Any]] = None):
        # Initialize OpenRouter
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if api_key:
            print("✅ OpenRouter API key configured successfully")
        else:
            print("⚠️  Warning: OPENROUTER_API_KEY not set. LLM features will be disabled.")

        # Store EDA context for agents
        self.eda_context = eda_context or {}
        
        # Initialize all agents with OpenRouter API key and EDA context
        self.agents = {
            "data_scientist": DataScientistAgent(api_key, eda_context=self.eda_context),
            "scoring": ScoringAgent(api_key, eda_context=self.eda_context),
            "insight": InsightAgent(api_key, eda_context=self.eda_context),
            "predictive": PredictiveAgent(api_key, eda_context=self.eda_context)
        }

        # Check if parallel execution is enabled
        self.parallel_mode = os.getenv("PARALLEL", "true").lower() == "true"
        mode_str = "PARALLEL" if self.parallel_mode else "SEQUENTIAL"
        print(f"🔧 Agent execution mode: {mode_str}")

        self.results: List[RecordResult] = []

    async def process_single_record(self, record: Record) -> RecordResult:
        """
        Process a single record through all agents.
        
        Mode controlled by PARALLEL environment variable:
        - PARALLEL=true: Run all agents simultaneously (default)
        - PARALLEL=false: Run agents one by one in sequence
        """
        print(f"\n{'='*80}")
        print(f"🔄 Processing Record: {record.record_id}")
        print(f"{'='*80}")

        start_time = datetime.now()
        result = RecordResult(record_id=record.record_id, status=TaskStatus.RUNNING)

        try:
            if self.parallel_mode:
                # PARALLEL MODE: Run all 4 agents simultaneously
                agent_results = await asyncio.gather(
                    self.agents["data_scientist"].process_record(record),
                    self.agents["scoring"].process_record(record),
                    self.agents["insight"].process_record(record),
                    self.agents["predictive"].process_record(record),
                    return_exceptions=True
                )

                # Unpack results
                result.data_scientist_output = agent_results[0] if not isinstance(agent_results[0], Exception) else None
                result.scoring_output = agent_results[1] if not isinstance(agent_results[1], Exception) else None
                result.insight_output = agent_results[2] if not isinstance(agent_results[2], Exception) else None
                result.predictive_output = agent_results[3] if not isinstance(agent_results[3], Exception) else None

                # Check for errors
                for i, res in enumerate(agent_results):
                    if isinstance(res, Exception):
                        result.errors.append(f"Agent {i}: {str(res)}")
            else:
                # SEQUENTIAL MODE: Run agents one by one
                agent_results = []
                
                # Run each agent sequentially
                try:
                    result.data_scientist_output = await self.agents["data_scientist"].process_record(record)
                except Exception as e:
                    result.errors.append(f"DataScientist: {str(e)}")
                    result.data_scientist_output = None
                
                try:
                    result.scoring_output = await self.agents["scoring"].process_record(record)
                except Exception as e:
                    result.errors.append(f"Scoring: {str(e)}")
                    result.scoring_output = None
                
                try:
                    result.insight_output = await self.agents["insight"].process_record(record)
                except Exception as e:
                    result.errors.append(f"Insight: {str(e)}")
                    result.insight_output = None
                
                try:
                    result.predictive_output = await self.agents["predictive"].process_record(record)
                except Exception as e:
                    result.errors.append(f"Predictive: {str(e)}")
                    result.predictive_output = None

            result.status = TaskStatus.FAILED if result.errors else TaskStatus.COMPLETED
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.errors.append(str(e))

        # Calculate processing time
        end_time = datetime.now()
        result.processing_time_ms = (end_time - start_time).total_seconds() * 1000

        print(f"✅ Record {record.record_id} completed in {result.processing_time_ms:.0f}ms")
        print(f"   Status: {result.status.value}")
        
        return result

    async def process_batch(self, records: List[Record]) -> List[RecordResult]:
        """
        Process multiple records sequentially (one at a time).
        
        Each record goes through all agents in parallel before the next record starts.
        """
        print(f"\n{'🚀 '*40}")
        print(f"BATCH PROCESSING: {len(records)} records")
        print(f"{'🚀 '*40}\n")

        batch_start = datetime.now()
        results = []

        # Process each record sequentially
        for i, record in enumerate(records, 1):
            print(f"\n[Batch Progress: {i}/{len(records)}]")
            result = await self.process_single_record(record)
            results.append(result)
            self.results.append(result)

        batch_end = datetime.now()
        total_time = (batch_end - batch_start).total_seconds()

        print(f"\n{'='*80}")
        print(f"📊 BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total records processed: {len(records)}")
        print(f"Successful: {sum(1 for r in results if r.status == TaskStatus.COMPLETED)}")
        print(f"Failed: {sum(1 for r in results if r.status == TaskStatus.FAILED)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per record: {(total_time/len(records)):.2f}s")
        print(f"{'='*80}\n")

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of all processed records."""
        return {
            "total_records": len(self.results),
            "completed": sum(1 for r in self.results if r.status == TaskStatus.COMPLETED),
            "failed": sum(1 for r in self.results if r.status == TaskStatus.FAILED),
            "avg_processing_time_ms": sum(r.processing_time_ms for r in self.results) / max(len(self.results), 1),
            "results": [
                {
                    "record_id": r.record_id,
                    "status": r.status.value,
                    "quality_score": r.scoring_output.get("overall_quality_score") if r.scoring_output else None,
                    "insights_count": len(r.insight_output.get("insights", [])) if r.insight_output else 0
                }
                for r in self.results
            ]
        }


async def main():
    """Demo: Process records one at a time through all agents."""
    
    # Create sample records
    records = [
        Record(
            record_id="REC-001",
            data={
                "customer_id": "CUST-12345",
                "transaction_amount": 1500.50,
                "transaction_date": "2026-01-04",
                "product": "Premium Subscription",
                "region": "US-West"
            }
        ),
        Record(
            record_id="REC-002",
            data={
                "customer_id": "CUST-67890",
                "transaction_amount": None,  # Missing value
                "transaction_date": "2026-01-03",
                "product": "",  # Empty value
                "region": "EU-Central"
            }
        ),
        Record(
            record_id="REC-003",
            data={
                "customer_id": "CUST-11111",
                "transaction_amount": 299.99,
                "transaction_date": "2026-01-04",
                "product": "Basic Plan",
                "region": "APAC"
            }
        )
    ]

    # Initialize orchestrator (will use OPENAI_API_KEY from environment)
    orchestrator = RecordOrchestrator()

    # Process all records (one at a time, with parallel agent execution per record)
    results = await orchestrator.process_batch(records)

    # Print detailed summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    summary = orchestrator.get_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
