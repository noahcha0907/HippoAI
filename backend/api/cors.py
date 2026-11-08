"""Which web pages may call the API from a browser. Shared by the mock and real APIs.

The public demo site (GitHub Pages) can call the API running on the presenter's laptop
during a live demo. Chrome requires the private-network header for a public page to reach
localhost. Add more origins with HIPPO_ALLOWED_ORIGINS (comma-separated, no paths).
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:4173",  # The static demo site (python -m http.server 4173).
    "https://noahcha0907.github.io",
]


def allowed_origins() -> list[str]:
    extra = [o.strip().rstrip("/") for o in os.environ.get("HIPPO_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return DEFAULT_ORIGINS + [o for o in extra if o not in DEFAULT_ORIGINS]


def add_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_private_network=True,
    )
