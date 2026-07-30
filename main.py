"""Cloud Run / local WSGI-less entrypoint for uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, proxy_headers=True)


if __name__ == "__main__":
    main()
