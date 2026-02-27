# ./tests/__init__.py
"""
Mark the tests directory as a package for stable import resolution.
Used by test modules importing shared helpers through `tests.test_helper`.
Run: Loaded implicitly by Python test runners.
Inputs: None.
Outputs: Package initialization side effects limited to namespace registration.
Side effects: None.
Operational notes: Kept intentionally minimal for deterministic test startup.
"""
