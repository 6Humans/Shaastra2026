# 🚀 Quick Start Guide

## What This Does

Processes data records **one at a time** through **4 AI agents running in parallel**:

```
Record 1 → [DataScientist + Scoring + Insight + Predictive] → ✅
Record 2 → [DataScientist + Scoring + Insight + Predictive] → ✅
Record 3 → [DataScientist + Scoring + Insight + Predictive] → ✅
```

## Installation

```bash
# Clone and setup
cd iit
uv sync

# Add your Gemini key to .env
echo "GEMINI_API_KEY=your-key-here" > .env
```

## Run It

```bash
# Basic demo (works without Gemini)
uv run python main.py

# With Gemini insights
uv run python demo_with_gemini.py
```

## Expected Output

```
🔄 Processing Record: REC-001
[DataScientistAgent] Processing record REC-001
[ScoringAgent] Scoring record REC-001
[InsightAgent] Generating insights for record REC-001
[PredictiveAgent] Forecasting for record REC-001
✅ Record REC-001 completed in 302ms

🔄 Processing Record: REC-002
[DataScientistAgent] Processing record REC-002
...
```

## Files Overview

| File | Purpose |
|------|---------||
| `main.py` | Main entry point - runs record processing demo |
| `src/record_processor.py` | Core system with all 4 agents |
| `src/agent.py` | Original batch-style agents (alternative) |
| `src/colab_integration.py` | SSH/Colab remote execution template |
| `demo_with_gemini.py` | Quick Gemini integration test |

## Architecture

### 🎭 RecordOrchestrator
- Processes records sequentially
- Runs 4 agents in parallel per record
- Tracks results and timing

### 🤖 The 4 Agents

1. **DataScientistAgent** - Validates data, checks completeness
2. **ScoringAgent** - Computes quality scores (0-1 scale)
3. **InsightAgent** - Uses Gemini to generate insights
4. **PredictiveAgent** - Uses Gemini to forecast issues

## Custom Usage

```python
from src.record_processor import RecordOrchestrator, Record

# Create your records
records = [
    Record(
        record_id="YOUR-001",
        data={
            "field1": "value1",
            "field2": 123,
            "field3": None  # Missing value detected!
        }
    )
]

# Process them
orchestrator = RecordOrchestrator()
results = await orchestrator.process_batch(records)

# Check results
for result in results:
    print(f"{result.record_id}: {result.scoring_output['overall_quality_score']}")
```

## Google Gemini Integration

When `GEMINI_API_KEY` is set:
- **InsightAgent** calls Gemini 1.5 Flash for human-readable insights
- **PredictiveAgent** uses Gemini to forecast issues
- **DataScientistAgent** gets AI-powered quality assessment

Without Gemini:
- System still works with basic scoring
- No LLM-generated insights
- Faster but less intelligent

## Performance

- **Without Gemini**: ~300ms per record
- **With Gemini**: ~1-2s per record (due to API calls)
- **Parallelism**: 4 agents run simultaneously per record

## Colab Integration (Optional)

See `src/colab_integration.py` for:
- Running heavy computations on Colab GPU
- SSH-based remote execution
- Offloading model training to cloud

## Next Steps

1. ✅ Run `uv run python main.py` to test basic flow
2. 🔑 Add Gemini key to `.env` for AI insights
3. 📝 Modify `main.py` with your own records
4. 🚀 Deploy with `src/colab_integration.py` for scale

## Troubleshooting

**"GEMINI_API_KEY not set"**
- Create `.env` file with your key
- Or set environment variable: `export GEMINI_API_KEY=your-key`

**Slow processing**
- Expected with Gemini calls (~1-2s per record)
- Disable Gemini for faster testing (300ms per record)

**Import errors**
- Run `uv sync` to install dependencies
- Make sure you're in the project directory

## License

MIT
