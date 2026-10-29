"""API load testing script with Locust.

Usage:
    locust -f scripts/load_test.py --host http://localhost:8000

Optional quick smoke run without the web UI:
    python scripts/load_test.py --base-url http://localhost:8000 --requests 100
"""

from __future__ import annotations

import argparse
import random
import time

import requests

try:
    from locust import HttpUser, between, task
except Exception:  # pragma: no cover
    HttpUser = object  # type: ignore[assignment]

    def task(weight: int = 1):
        """Fallback task decorator when locust is unavailable."""

        def _inner(func):
            return func

        del weight
        return _inner

    def between(_min_wait: float, _max_wait: float):
        """Fallback wait-time callable when locust is unavailable."""

        def _wait() -> float:
            return 0.0

        return _wait


class RetailApiUser(HttpUser):
    """Locust user profile for core API traffic patterns."""

    wait_time = between(0.2, 1.0)

    @task(4)
    def health(self) -> None:
        """Hit health endpoint."""
        self.client.get("/health", name="GET /health")

    @task(3)
    def kpis(self) -> None:
        """Hit KPI endpoint."""
        self.client.get("/api/metrics/kpis", name="GET /api/metrics/kpis")

    @task(2)
    def forecast(self) -> None:
        """Hit forecast endpoint with random store selection."""
        payload = {"store_id": random.randint(1, 1115), "horizon": 7}
        self.client.post("/api/forecast/", json=payload, name="POST /api/forecast/")

    @task(1)
    def alerts(self) -> None:
        """Hit alerts endpoint."""
        self.client.get("/api/alerts/active", name="GET /api/alerts/active")


def quick_run(base_url: str, requests_count: int = 100, use_testclient: bool = False) -> int:
    """Run a minimal synchronous load probe without Locust UI.

    Args:
        base_url: API base URL.
        requests_count: Number of requests to issue.
        use_testclient: Whether to use FastAPI TestClient instead of requests.

    Returns:
        Process exit code.
    """
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/api/metrics/kpis", None),
        ("POST", "/api/v1/forecast/", {"store_id": 1, "horizon": 14}),
    ]

    per_endpoint_stats: dict[str, dict] = {}

    if use_testclient:
        import sys
        from pathlib import Path

        from fastapi.testclient import TestClient

        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        from src.api.main import app

        session = TestClient(app)
    else:
        session = requests.Session()

    for method, path, payload in endpoints:
        latencies: list[float] = []
        failures = 0
        errors: list[str] = []
        for _ in range(int(requests_count)):
            t0 = time.perf_counter()
            try:
                if use_testclient:
                    if method == "GET":
                        response = session.get(path)
                    else:
                        response = session.post(path, json=payload)
                else:
                    if method == "GET":
                        response = session.get(f"{base_url}{path}", timeout=15)
                    else:
                        response = session.post(f"{base_url}{path}", json=payload, timeout=15)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)
                if response.status_code >= 500:
                    failures += 1
                    errors.append(f"HTTP {response.status_code}")
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)
                failures += 1
                errors.append(str(exc))

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        p99_idx = int(n * 0.99) - 1
        p99_idx = max(0, p99_idx)

        per_endpoint_stats[path] = {
            "method": method,
            "requests": n,
            "failures": failures,
            "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
            "median_latency_ms": sorted_latencies[n // 2] if n else 0.0,
            "p99_latency_ms": sorted_latencies[p99_idx] if n else 0.0,
            "min_latency_ms": min(latencies) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "errors": errors[:5],
            "error_count": len(errors),
        }

    # Build report
    lines = []
    lines.append("=" * 60)
    lines.append("API LOAD TEST RESULTS")
    lines.append("=" * 60)
    lines.append(f"Base URL: {base_url}")
    lines.append(f"Requests per endpoint: {requests_count}")
    lines.append("")

    all_pass = True
    for path, stats in per_endpoint_stats.items():
        lines.append(f"Endpoint: {stats['method']} {path}")
        lines.append(f"  Requests:       {stats['requests']}")
        lines.append(f"  Mean latency:   {stats['avg_latency_ms']:.2f} ms")
        lines.append(f"  Median latency: {stats['median_latency_ms']:.2f} ms")
        lines.append(f"  P99 latency:    {stats['p99_latency_ms']:.2f} ms")
        lines.append(f"  Min latency:    {stats['min_latency_ms']:.2f} ms")
        lines.append(f"  Max latency:    {stats['max_latency_ms']:.2f} ms")
        lines.append(f"  Failures:       {stats['failures']}")
        if stats["errors"]:
            for err in stats["errors"]:
                lines.append(f"    - {err}")
            if stats["error_count"] > 5:
                lines.append(f"    ... and {stats['error_count'] - 5} more")
        lines.append("")

        # Check criteria
        p99_ok = stats["p99_latency_ms"] < 2000
        no_5xx = stats["failures"] == 0
        lines.append(f"  P99 < 2s:       {'PASS' if p99_ok else 'FAIL'}")
        lines.append(f"  No 5xx errors:  {'PASS' if no_5xx else 'FAIL'}")
        lines.append("")

        if not p99_ok or not no_5xx:
            all_pass = False

    lines.append("=" * 60)
    lines.append(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    # Save to evidence file
    evidence_path = Path(".sisyphus/evidence/t13-load-test.log")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nResults saved to: {evidence_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument(
        "--use-testclient",
        action="store_true",
        help="Use FastAPI TestClient instead of HTTP requests",
    )
    args = parser.parse_args()
    raise SystemExit(
        quick_run(
            base_url=args.base_url, requests_count=args.requests, use_testclient=args.use_testclient
        )
    )
