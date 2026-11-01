import uvicorn
import os
import sys

# Add backend directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

if __name__ == "__main__":
    print("Starting NeuroAdapt Backend...")
    # Run FastAPI app using uvicorn
    # The reload=True allows auto-reloading when code changes (great for development)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
