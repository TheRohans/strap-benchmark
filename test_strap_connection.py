#!/usr/bin/env python3
"""
Smoke test: connect to Strap, say "test", verify it responds within 60s.

Usage:
    python test_strap_connection.py
    STRAP_HOST=192.168.1.36 STRAP_PASSWORD=secret python test_strap_connection.py
"""
import sys
from runners.strap import StrapRunner


def test_strap_responds():
    runner = StrapRunner(timeout_secs=60.0)
    print(f"Connecting to {runner.host}:{runner.port} on {runner.channel} ...")

    runner.connect()

    task = {"id": "smoke", "type": "reasoning", "prompt": "test"}
    response = runner.run(task)
    runner.close()

    assert response.strip(), "No response received within timeout"
    print(f"OK — got response ({len(response)} chars):\n{response[:300]}")


if __name__ == "__main__":
    try:
        test_strap_responds()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
