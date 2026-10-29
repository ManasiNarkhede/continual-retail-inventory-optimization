"""Simple backend reachability checker using requests.

Usage:
    .venv39\Scripts\python scripts/check_backend.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys

import requests


def main(base_url: str, timeout: float) -> int:
    """Ping backend health endpoint and print strict result.

    Args:
        base_url: Base URL for backend service.
        timeout: Request timeout in seconds.

    Returns:
        Exit code, where 0 means reachable and healthy.
    """
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        print(f"CHECK_FAIL request_error={exc}")
        return 1

    if response.status_code != 200:
        print(f"CHECK_FAIL status_code={response.status_code} body={response.text[:300]}")
        return 2

    print(f"CHECK_OK status_code={response.status_code} body={response.text[:300]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    sys.exit(main(base_url=args.base_url, timeout=args.timeout))
