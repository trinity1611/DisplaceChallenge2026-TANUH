"""
DISPLACE MedAI – Server Launcher
===================================
Run this file to start the microservice:

    python -m backend.run

Or:

    python backend/run.py
"""

import uvicorn
from backend.app.config import settings


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║            DISPLACE MedAI Microservice                ║")
    print("║        Medical Audio Analysis Pipeline                ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print(f"  🌐 Server:  http://localhost:{settings.port}")
    print(f"  📖 API Docs: http://localhost:{settings.port}/docs")
    print(f"  🔧 Device:   {settings.device}")
    print()

    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
