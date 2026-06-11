"""
Smoke test for the AIActChecker FastAPI service.

Prerequisites: the API must be running. In one terminal:
    uvicorn src.api:app --reload

Then run this script in another terminal:
    python scripts/test_api.py

This test exercises:
  - GET /        (root info)
  - GET /health  (liveness)
  - POST /classify  (Mode 1, 1 LLM call)
  - POST /cross-map (Mode 4, 1 LLM call)

Total LLM calls: 2. Other modes (obligations, gap analysis) are tested via the
Swagger UI at http://localhost:8000/docs to avoid extra rate limit pressure
during automated tests.
"""

import sys
import textwrap

import requests

BASE_URL = "http://127.0.0.1:8000"
SEPARATOR = "-" * 78


def step(label, fn):
    print(f"\n{label}")
    try:
        fn()
        print(f"  [PASS]")
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] Could not reach {BASE_URL}. Is uvicorn running?")
        sys.exit(1)


def test_root():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"unexpected status {r.status_code}"
    data = r.json()
    assert data.get("name") == "AIActChecker API"
    print(f"  Version: {data.get('version')}")
    print(f"  Modes exposed: {len(data.get('modes', {}))}")


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_classify():
    payload = {
        "system_description": "An AI tool that screens job applicants by scoring CVs"
    }
    r = requests.post(f"{BASE_URL}/classify", json=payload, timeout=120)
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"
    data = r.json()
    risk = data.get("risk_category")
    assert risk == "HIGH", f"expected HIGH, got {risk}"
    print(f"  Risk category: {risk}")
    refs = data.get("article_references", [])
    if refs:
        print(f"  Citations: {', '.join(refs[:3])}")


def test_cross_map():
    payload = {"topic": "Human Oversight of AI Systems"}
    r = requests.post(f"{BASE_URL}/cross-map", json=payload, timeout=120)
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"
    data = r.json()
    mappings = data.get("mappings", {})
    assert "eu_ai_act" in mappings
    assert "gdpr" in mappings
    assert "nist_ai_rmf" in mappings
    print(f"  Topic: {data.get('topic', '?')}")
    eu_refs = mappings.get("eu_ai_act", {}).get("primary_references", [])
    if eu_refs:
        print(f"  EU AI Act refs: {', '.join(eu_refs[:2])}")


def main():
    print(SEPARATOR)
    print("AIActChecker FastAPI smoke test")
    print(f"Target: {BASE_URL}")
    print(SEPARATOR)

    step("GET /        (root info)", test_root)
    step("GET /health  (liveness)", test_health)
    step("POST /classify  (Mode 1)", test_classify)
    step("POST /cross-map (Mode 4)", test_cross_map)

    print(f"\n{SEPARATOR}")
    print("All smoke tests passed.")
    print(f"Open http://localhost:8000/docs to try /obligations, /gap-analysis,")
    print(f"and /query interactively via Swagger UI.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
