"""
SSH/Colab Integration Example for Remote Agent Execution.

This shows how to execute agents on remote Colab instances via SSH.
"""

import asyncio
import paramiko
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ColabConnection:
    """Configuration for Colab SSH connection."""
    host: str
    port: int = 22
    username: str = "root"
    password: Optional[str] = None
    key_filename: Optional[str] = None


class RemoteAgentExecutor:
    """
    Executes agent code on remote Colab instance via SSH.
    
    This allows heavy computation (data processing, model training)
    to run on Colab's GPU/TPU resources.
    """

    def __init__(self, connection: ColabConnection):
        self.connection = connection
        self.ssh_client: Optional[paramiko.SSHClient] = None

    async def connect(self):
        """Establish SSH connection to Colab."""
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to Colab via SSH
        self.ssh_client.connect(
            hostname=self.connection.host,
            port=self.connection.port,
            username=self.connection.username,
            password=self.connection.password,
            key_filename=self.connection.key_filename
        )
        print(f"✅ Connected to Colab at {self.connection.host}")

    async def execute_remote_code(self, code: str) -> str:
        """Execute Python code on remote Colab instance."""
        if not self.ssh_client:
            raise ConnectionError("Not connected to Colab. Call connect() first.")
        
        # Create a temporary Python file
        temp_file = "/tmp/agent_code.py"
        
        # Upload code to Colab
        stdin, stdout, stderr = self.ssh_client.exec_command(
            f'cat > {temp_file} << "EOF"\n{code}\nEOF'
        )
        
        # Execute the code
        stdin, stdout, stderr = self.ssh_client.exec_command(f'python3 {temp_file}')
        
        # Get output
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if error:
            print(f"⚠️  Stderr: {error}")
        
        return output

    async def run_data_science_task(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run data science tasks on Colab."""
        code = f"""
import json
import pandas as pd
import numpy as np

# Record data
data = {record_data}

# Perform EDA
df = pd.DataFrame([data])
result = {{
    "shape": list(df.shape),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "describe": df.describe().to_dict(),
    "missing": df.isnull().sum().to_dict()
}}

print(json.dumps(result))
"""
        output = await self.execute_remote_code(code)
        
        try:
            return json.loads(output)
        except:
            return {"output": output}

    async def run_model_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run predictive model training on Colab GPU."""
        code = f"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simulated model training
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

result = {{
    "model_score": float(model.score(X, y)),
    "feature_importance": model.feature_importances_.tolist(),
    "n_estimators": model.n_estimators
}}

print(json.dumps(result))
"""
        output = await self.execute_remote_code(code)
        
        try:
            return json.loads(output)
        except:
            return {"output": output}

    async def disconnect(self):
        """Close SSH connection."""
        if self.ssh_client:
            self.ssh_client.close()
            print("✅ Disconnected from Colab")


# Example integration with existing agents
class RemoteDataScientistAgent:
    """DataScientist agent that executes on Colab via SSH."""

    def __init__(self, colab_connection: ColabConnection):
        self.remote_executor = RemoteAgentExecutor(colab_connection)
        self.name = "RemoteDataScientistAgent"

    async def process_record(self, record) -> Dict[str, Any]:
        """Process record on remote Colab instance."""
        await self.remote_executor.connect()
        
        try:
            # Run EDA on Colab
            result = await self.remote_executor.run_data_science_task(record.data)
            return result
        finally:
            await self.remote_executor.disconnect()


class RemotePredictiveAgent:
    """Predictive agent that trains models on Colab GPU."""

    def __init__(self, colab_connection: ColabConnection):
        self.remote_executor = RemoteAgentExecutor(colab_connection)
        self.name = "RemotePredictiveAgent"

    async def process_record(self, record) -> Dict[str, Any]:
        """Train predictive model on Colab."""
        await self.remote_executor.connect()
        
        try:
            # Run model training on Colab GPU
            result = await self.remote_executor.run_model_training(record.data)
            return result
        finally:
            await self.remote_executor.disconnect()


# Usage example
async def demo_remote_execution():
    """
    Demo: How to use Colab for remote agent execution.
    
    Setup Colab SSH:
    1. In Colab notebook, run:
       !pip install colab-ssh
       from colab_ssh import launch_ssh
       launch_ssh('your_password', 'your_ngrok_token')
    
    2. Get the SSH connection details (host, port)
    3. Use them here
    """
    
    # Example connection (replace with your Colab SSH details)
    colab_conn = ColabConnection(
        host="0.tcp.ngrok.io",  # From Colab ngrok tunnel
        port=12345,              # From Colab ngrok tunnel
        username="root",
        password="your_colab_password"
    )
    
    # Create remote agents
    remote_ds_agent = RemoteDataScientistAgent(colab_conn)
    remote_pred_agent = RemotePredictiveAgent(colab_conn)
    
    # Create a test record
    from src.record_processor import Record
    
    test_record = Record(
        record_id="REMOTE-001",
        data={
            "value1": 100,
            "value2": 200,
            "value3": 300
        }
    )
    
    print("🚀 Executing agents on Colab...")
    
    # Run agents remotely in parallel
    results = await asyncio.gather(
        remote_ds_agent.process_record(test_record),
        remote_pred_agent.process_record(test_record)
    )
    
    print("\n✅ Remote execution complete!")
    print(f"DataScience result: {results[0]}")
    print(f"Predictive result: {results[1]}")


if __name__ == "__main__":
    print("""
    📝 SSH/Colab Integration Guide
    ================================
    
    This module shows how to execute agents on Google Colab via SSH.
    
    Benefits:
    - Run heavy computations on Colab's GPU/TPU
    - Scale processing without local resources
    - Keep data processing isolated
    
    Setup:
    1. Install: uv add paramiko
    2. Configure Colab SSH (see demo_remote_execution docstring)
    3. Update ColabConnection with your details
    4. Run: python src/colab_integration.py
    
    Note: This is a template. Uncomment demo to test.
    """)
    
    # Uncomment to run demo (after setting up Colab SSH):
    # asyncio.run(demo_remote_execution())
