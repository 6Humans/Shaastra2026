#!/usr/bin/env python3
"""
Quick demo to test Google Gemini integration.
Run: python demo_with_gemini.py
"""

import asyncio
import os
from src.record_processor import RecordOrchestrator, Record


async def main():
    # You can either:
    # 1. Set GEMINI_API_KEY in .env file, OR
    # 2. Pass it directly here (for testing)
    
    # Example: Set it directly (REMOVE THIS BEFORE COMMITTING!)
    # os.environ['GEMINI_API_KEY'] = 'your-key-here'
    
    # Create a test record
    test_record = Record(
        record_id="TEST-001",
        data={
            "customer_id": "CUST-999",
            "transaction_amount": 999.99,
            "transaction_date": "2026-01-04",
            "product": "Test Product",
            "region": "TEST"
        }
    )
    
    # Initialize orchestrator
    orchestrator = RecordOrchestrator()
    
    # Process single record
    print("\n" + "="*80)
    print("🧪 TESTING SINGLE RECORD WITH OPENAI")
    print("="*80)
    
    result = await orchestrator.process_single_record(test_record)
    
    print("\n" + "="*80)
    print("📊 RESULT")
    print("="*80)
    print(f"Record ID: {result.record_id}")
    print(f"Status: {result.status.value}")
    print(f"Processing Time: {result.processing_time_ms:.0f}ms")
    
    if result.insight_output:
        print("\n💡 INSIGHTS:")
        for insight in result.insight_output.get('insights', []):
            print(f"  • {insight}")
        
        print("\n⚠️  RISKS:")
        for risk in result.insight_output.get('risks', []):
            print(f"  • {risk}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
