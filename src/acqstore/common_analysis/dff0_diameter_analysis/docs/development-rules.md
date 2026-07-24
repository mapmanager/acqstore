# Development Rules

1. Keep the generic core domain-independent.
2. Use adapters at AcqStore and sidecar boundaries.
3. Use sample indices internally; convert display units at boundaries.
4. Do not redetect supplied seeds.
5. Return one attempted structured result per seed.
6. Truncate each event at the next seed or signal end.
7. Preserve raw data; filtering always creates derived values.
8. Represent unresolved measurements explicitly with status, warnings, and `None`.
9. Design stable extension points without implementing speculative extensions.
10. Before adding a materially deeper algorithm, abstraction, parameter family, or preprocessing stage, provide a recommendation and rationale and obtain explicit human approval.
11. Prefer a complete simple measurement over a partially validated complex detector.
12. Separate measurement from biological interpretation.
13. Use Google-style docstrings at module, class, and function levels where appropriate.
14. Add focused comments for non-obvious algorithmic code.
15. Use explicit type annotations and frozen dataclasses for parameters and serialized results.
16. Provide `to_dict()` and `from_dict()` for schema-backed models.
17. Keep NiceGUI imports isolated under `app/`; analysis and plotting remain independently usable.
18. Maintain tests, docs, plotting, and the working app as first-class parts of the package.
19. ChatGPT acts as a senior developer on this analysis project. When clarification is required, ask rather than guess and provide the expert recommended answer first, with rationale proportional to complexity.
