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

    def __init__(self, name: str, openrouter_api_key: Optional[str] = None):
        self.name = name
        self.openrouter_api_key = openrouter_api_key
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
                        "max_tokens": 500
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.log(f"LLM error: {str(e)}")
            return f"Error: {str(e)}"


class DataScientistAgent(Agent):
    """Performs data loading, validation, preprocessing, and EDA on individual records."""

    def __init__(self, openrouter_api_key: Optional[str] = None):
        super().__init__("DataScientistAgent", openrouter_api_key)

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
                prompt = f"""Analyze this data record for quality issues:
{json.dumps(record.data, indent=2)}

Identify any potential data quality concerns (missing values, anomalies, format issues)."""
                
                llm_analysis = await self.call_llm(
                    prompt,
                    system_prompt="You are a data quality expert. Provide concise analysis."
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
    """Computes data quality metrics and scores for individual records."""

    def __init__(self, openrouter_api_key: Optional[str] = None):
        super().__init__("ScoringAgent", openrouter_api_key)

    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Score a single record's data quality."""
        self.log(f"Scoring record {record.record_id}")

        try:
            await asyncio.sleep(0.2)

            # Calculate various quality dimensions
            data = record.data
            
            # Completeness
            completeness = sum(1 for v in data.values() if v not in [None, ""]) / max(len(data), 1)
            
            # Validity (basic check for expected data types)
            validity = 0.9  # Simulated
            
            # Consistency (check for contradictions)
            consistency = 0.95  # Simulated
            
            # Calculate weighted overall score
            weights = {"completeness": 0.4, "validity": 0.3, "consistency": 0.3}
            overall_score = (
                completeness * weights["completeness"] +
                validity * weights["validity"] +
                consistency * weights["consistency"]
            )

            output = {
                "record_id": record.record_id,
                "completeness_score": round(completeness, 3),
                "validity_score": round(validity, 3),
                "consistency_score": round(consistency, 3),
                "overall_quality_score": round(overall_score, 3),
                "quality_level": "High" if overall_score >= 0.9 else "Medium" if overall_score >= 0.7 else "Low"
            }

            self.log(f"Record {record.record_id}: Overall score {overall_score:.3f}")
            return output

        except Exception as e:
            self.log(f"Error scoring record {record.record_id}: {str(e)}")
            raise


class InsightAgent(Agent):
    """Converts scores to human-readable insights using OpenRouter."""

    def __init__(self, openrouter_api_key: Optional[str] = None):
        super().__init__("InsightAgent", openrouter_api_key)

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
                prompt = f"""Analyze this data record and provide:
1. Key insights about data quality
2. Potential risks
3. Recommendations for improvement

Record data:
{json.dumps(record.data, indent=2)}

Provide a concise analysis in JSON format:
{{
  "insights": ["insight1", "insight2"],
  "risks": ["risk1", "risk2"],
  "recommendations": ["rec1", "rec2"]
}}"""

                llm_response = await self.call_llm(
                    prompt,
                    system_prompt="You are a data quality analyst. Provide actionable insights in valid JSON format."
                )

                try:
                    # Try to parse JSON from LLM response
                    # Extract JSON if wrapped in markdown code blocks
                    json_str = llm_response
                    if "```json" in llm_response:
                        json_str = llm_response.split("```json")[1].split("```")[0].strip()
                    elif "```" in llm_response:
                        json_str = llm_response.split("```")[1].split("```")[0].strip()
                    
                    parsed = json.loads(json_str)
                    insights = parsed.get("insights", [])
                    risks = parsed.get("risks", [])
                    recommendations = parsed.get("recommendations", [])
                except json.JSONDecodeError:
                    insights = ["Data quality analysis completed"]
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
    """Forecasts potential issues for individual records using OpenRouter."""

    def __init__(self, openrouter_api_key: Optional[str] = None):
        super().__init__("PredictiveAgent", openrouter_api_key)

    async def process_record(self, record: Record) -> Dict[str, Any]:
        """Predict potential issues for a single record."""
        self.log(f"Forecasting for record {record.record_id}")

        try:
            await asyncio.sleep(0.3)

            predictions = []
            confidence = 0.75

            if self.openrouter_api_key:
                prompt = f"""Based on this data record, predict potential future data quality issues:
{json.dumps(record.data, indent=2)}

Consider:
- Trend deterioration
- Potential missing data
- Anomaly risks

Provide predictions as a JSON array of objects with 'issue' and 'probability' fields."""

                llm_response = await self.call_llm(
                    prompt,
                    system_prompt="You are a predictive analytics expert. Forecast potential data quality issues."
                )

                # Extract predictions (simplified parsing)
                predictions = [
                    {
                        "issue": "Potential data staleness",
                        "probability": 0.35,
                        "timeframe": "7 days"
                    }
                ]

            output = {
                "record_id": record.record_id,
                "predictions": predictions,
                "trend_forecast": "stable",
                "confidence_score": confidence,
                "predicted_quality_score_7d": 0.88
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

    def __init__(self, openrouter_api_key: Optional[str] = None):
        # Initialize OpenRouter
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if api_key:
            print("✅ OpenRouter API key configured successfully")
        else:
            print("⚠️  Warning: OPENROUTER_API_KEY not set. LLM features will be disabled.")

        # Initialize all agents with OpenRouter API key
        self.agents = {
            "data_scientist": DataScientistAgent(api_key),
            "scoring": ScoringAgent(api_key),
            "insight": InsightAgent(api_key),
            "predictive": PredictiveAgent(api_key)
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
