from pydantic import BaseModel, Field

class EdgeObservation(BaseModel):
    ram_usage: float = Field(..., description="Current RAM usage percentage (0.0 to 100.0)")
    battery_level: float = Field(..., description="Current battery percentage (0.0 to 100.0)")
    latency: float = Field(..., description="Current network latency in ms")
    active_processes: int = Field(..., description="Number of active processes")

class EdgeAction(BaseModel):
    kill_process: bool = Field(default=False, description="Terminate a non-critical process to free RAM")
    throttle_cpu: bool = Field(default=False, description="Throttle CPU to save battery at the cost of latency")
    route_to_cloud: bool = Field(default=False, description="Offload processing to the cloud to free RAM, increasing latency")

class EdgeReward(BaseModel):
    # CRITICAL FIX: Reward must be strictly typed and expected to be bounded 0.0 - 1.0 in env.py
    score: float = Field(..., description="Reward score strictly between 0.0 and 1.0")