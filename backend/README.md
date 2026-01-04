# Record-Based Parallel Agent System with Google Gemini

A sophisticated multi-agent system for data quality analysis that processes records individually with Google Gemini-powered insights.

## 🎯 Key Architecture

**Record-by-Record Processing:**
- Each record flows through ALL agents in parallel
- Next record starts only after previous record completes
- Google Gemini LLM provides intelligent analysis and insights

### 🎭 Orchestrator (Mother Agent)
- Processes records sequentially (one at a time)
- Runs all 4 agents in parallel for each record
- Aggregates results and tracks progress

### 🤖 Specialized Agents

1. **DataScientistAgent**
   - Validates data completeness
   - Detects missing/empty fields
   - Uses Google Gemini for quality assessment

2. **ScoringAgent**
   - Computes quality metrics (completeness, validity, consistency)
   - Calculates weighted overall score
   - Assigns quality levels (High/Medium/Low)

3. **InsightAgent** 💡
   - Uses Google Gemini to generate human-readable insights
   - Identifies risks and recommendations
   - Provides actionable analysis

4. **PredictiveAgent** 🔮
   - Uses Google Gemini to forecast potential issues
   - Predicts quality trends
   - Estimates future quality scores

## Installation

```bash
# Using uv (recommended)
uv sync

# Or install dependencies
uv add google-generativeai langgraph pydantic python-dotenv
```

## Setup

1. Create `.env` file:
```bash
cp .env.example .env
```

2. Add your Google Gemini API key:
```bash
GEMINI_API_KEY=your-key-here
```

## Usage

### Run the Demo

```bash
uv run python main.py
```

### Process Custom Records

```python
from src.record_processor import RecordOrchestrator, Record

# Create records
records = [
    Record(
        record_id="REC-001",
        data={
            "customer_id": "CUST-001",
            "amount": 1500.50,
            "date": "2026-01-04"
        }
    ),
    Record(
        record_id="REC-002",
        data={
            "customer_id": "CUST-002",
            "amount": None,  # Missing value
            "date": "2026-01-03"
        }
    )
]

# Process records
orchestrator = RecordOrchestrator()
results = await orchestrator.process_batch(records)
```

## Features

✅ **Record-by-Record Processing** - Each record gets complete analysis  
✅ **Parallel Agent Execution** - All 4 agents run simultaneously per record  
✅ **Google Gemini Integration** - Gemini 1.5 Flash powered insights and predictions  
✅ **Quality Scoring** - Multi-dimensional quality metrics  
✅ **Error Handling** - Graceful degradation on failures  
✅ **Progress Tracking** - Real-time batch progress  

## Execution Flow

```
Record 1:
├── DataScientistAgent ─┐
├── ScoringAgent       ─┤ Run in Parallel
├── InsightAgent       ─┤
└── PredictiveAgent    ─┘
    ↓ Complete

Record 2:
├── DataScientistAgent ─┐
├── ScoringAgent       ─┤ Run in Parallel
├── InsightAgent       ─┤
└── PredictiveAgent    ─┘
    ↓ Complete

... continues for each record
```

## Output Format

```json
{
  "record_id": "REC-001",
  "status": "completed",
  "data_scientist_output": {
    "completeness_score": 1.0,
    "missing_fields": [],
    "llm_analysis": "Data quality is excellent..."
  },
  "scoring_output": {
    "overall_quality_score": 0.955,
    "quality_level": "High"
  },
  "insight_output": {
    "insights": ["..."],
    "risks": ["..."],
    "recommendations": ["..."]
  },
  "predictive_output": {
    "predictions": [...],
    "trend_forecast": "stable"
  }
}
```

## Performance

- ~300ms per record (with 4 agents in parallel)
- Scales linearly with number of records
- Gemini API calls add ~1-2s per record (when enabled)
