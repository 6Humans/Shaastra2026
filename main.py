"""
Main entry point for the record-based parallel agent orchestration system.

Processes records individually - each record flows through all agents in parallel
before moving to the next record.

Uses OpenRouter with Qwen model for intelligent analysis and insights.
"""

import asyncio
import os
from dotenv import load_dotenv
from src.record_processor import RecordOrchestrator, Record

# Load environment variables
load_dotenv()


async def main():
    """Process records one at a time through all agents."""
    
    print("\n" + "🚀 "*40)
    print("RECORD-BASED PARALLEL AGENT SYSTEM WITH OPENROUTER (QWEN)")
    print("🚀 "*40)

    # Create sample records
    records = [
        Record(
            record_id="REC-001",
            data={
                "customer_id": "CUST-12345",
                "transaction_amount": 1500.50,
                "transaction_date": "2026-01-04",
                "product": "Premium Subscription",
                "region": "US-West",
                "payment_method": "credit_card"
            },
            metadata={"source": "api", "batch": "2026-01-04"}
        ),
        Record(
            record_id="REC-002",
            data={
                "customer_id": "CUST-67890",
                "transaction_amount": None,  # Missing value
                "transaction_date": "2026-01-03",
                "product": "",  # Empty value
                "region": "EU-Central",
                "payment_method": "paypal"
            },
            metadata={"source": "import", "batch": "2026-01-03"}
        ),
        Record(
            record_id="REC-003",
            data={
                "customer_id": "CUST-11111",
                "transaction_amount": 299.99,
                "transaction_date": "2026-01-04",
                "product": "Basic Plan",
                "region": "APAC",
                "payment_method": "bank_transfer"
            },
            metadata={"source": "api", "batch": "2026-01-04"}
        ),
        Record(
            record_id="REC-004",
            data={
                "customer_id": "CUST-22222",
                "transaction_amount": 5000.00,
                "transaction_date": "2026-01-04",
                "product": "Enterprise License",
                "region": "US-East",
                "payment_method": "wire_transfer"
            },
            metadata={"source": "manual", "batch": "2026-01-04"}
        )
    ]

    # Initialize orchestrator (will use OPENROUTER_API_KEY from environment)
    orchestrator = RecordOrchestrator()

    # Process all records (one at a time, with parallel agent execution per record)
    results = await orchestrator.process_batch(records)

    # Print final summary
    print("\n" + "="*80)
    print("📊 FINAL PROCESSING SUMMARY")
    print("="*80)
    summary = orchestrator.get_summary()
    
    print(f"\nTotal Records: {summary['total_records']}")
    print(f"✅ Completed: {summary['completed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏱️  Avg Processing Time: {summary['avg_processing_time_ms']:.0f}ms per record")
    
    print("\n" + "-"*80)
    print("Individual Record Results:")
    print("-"*80)
    for res in summary['results']:
        status_icon = "✅" if res['status'] == 'completed' else "❌"
        quality = f"{res['quality_score']:.3f}" if res['quality_score'] else "N/A"
        print(f"{status_icon} {res['record_id']}: Quality={quality}, Insights={res['insights_count']}")
    
    print("\n" + "="*80)
    print("💡 EXECUTION PATTERN:")
    print("="*80)
    print("Each record processed through ALL 4 agents in parallel:")
    print("  1️⃣  Record 1 → [DataScientist, Scoring, Insight, Predictive] → Complete")
    print("  2️⃣  Record 2 → [DataScientist, Scoring, Insight, Predictive] → Complete")
    print("  3️⃣  Record 3 → [DataScientist, Scoring, Insight, Predictive] → Complete")
    print("  4️⃣  Record 4 → [DataScientist, Scoring, Insight, Predictive] → Complete")
    print("\n✨ OpenRouter (Qwen 3-235B) LLM used for intelligent analysis and insights!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
