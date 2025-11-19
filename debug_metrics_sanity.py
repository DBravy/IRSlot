"""
Small debug helper to sanity-check metric sanitization / JSON-safety.

Run with:
    python debug_metrics_sanity.py

It prints out a few raw metric values and their sanitized versions.
"""

from app import _sanitize_metric


def main():
    raw_values = [0.1234, 1.0, float('nan'), float('inf'), float('-inf'), None, "not-a-number"]
    print("Raw -> Sanitized")
    for v in raw_values:
        safe = _sanitize_metric(v)
        print(f"{repr(v):>12} -> {safe}")


if __name__ == "__main__":
    main()


