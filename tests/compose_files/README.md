# Fixtures for tests/test_compose_models.py

| File | Source | Commit | sha256 | Licence |
|---|---|---|---|---|
| `land_2017.ode` | odezoo `land_2017`, version `github-6e5a5aa`; ComputationalPhysiology/crossbridge `src/crossbridge/land2017.ode` | `6e5a5aa543a57d5ecfbae7af4bd7d4ed9f1bdbfd` | `84c84b4275f6b76f9fcb17fe47df7cdaa7679bed7fa63605391fe0c575aae267` | MIT |
| `tentusscher_2006_epi.cellml` | odezoo `tentusscher_2006_epi`, version `pmr-5dc4239`; Physiome Model Repository workspace `tentusscher_panfilov_2006`, `ten_tusscher_model_2006_IK1Ko_epi_units.cellml` | `5dc42395eef6044fe766786f7bff197dea355eb3` | `999e4af049776f18a35aa30e09c3a8fa9487ab0b2e9ef6a77489dd371e1d6e6a` | CC-BY-3.0 |

The ToR-ORd model is not copied here: the recipe uses the existing fixture
`tests/cellml_files/ToRORd_dynCl_mid.cellml`.

`toy_*` are the fixtures of the toy composition; `tt06_land_*` and `torord_land*` are the
recipes and glue files of the real models.
