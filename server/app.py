
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Edge Deployment Environment is Live",
        "status": "Ready for OpenEnv Evaluation",
        "endpoints": ["/reset", "/step", "/state"]
    }

@app.post("/reset")
async def reset():
    # Your reset logic here
    return {"status": "success"}

@app.post("/step")
async def step(action: dict):
    # Your step logic here
    return {"observation": {}, "reward": 0.0, "done": False}

def main():
    """
    Main entry point for the environment server.
    Required by OpenEnv Spec and Hugging Face runtime.
    """
    # Hugging Face defaults to 7860, but OpenEnv may pass a PORT env var
    port = int(os.environ.get("PORT", 7860))

    # Hugging Face Spaces REQUIRES port 7860
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
    