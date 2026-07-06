#!/usr/bin/env python3
"""Run the RAG API Gateway (new microservice architecture)."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "part2_rag.api_gateway.gateway:gateway_app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
