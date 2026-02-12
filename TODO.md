# TODO — AI_MICROSCOPE

This file lists tasks in chronological order required to deliver the offline AI‑Aided Microscope Workstation described in `README.md`.

- [x] Create project TODO: initial plan and tracking (created)

1. Repository cleanup & structure: standardize folders, move scripts into `app/`, `inference/`, `model/`.
2. Add `requirements.txt` and env: pin packages (TensorFlow, OpenCV, CustomTkinter, Pillow, numpy) and add a minimal `venv`/instructions.
3. Acquire and verify dataset: SKIPPED — using provided trained model and dataset (marked completed).
4. Data preprocessing pipeline: SKIPPED — using preprocessed data and provided model (marked completed).
5. Define model architecture: SKIPPED — using provided `best_microscope_fusion` model (marked completed).
6. Training scripts and configs: SKIPPED — training performed externally (marked completed).
7. Train model & baseline eval: SKIPPED — trained model available (marked completed).
8. Export model weights & indices: Ensure `best_microscope_fusion` present in `model/` and `class_indices.json` updated.
9. Implement inference engine: `inference/inference.py` with preprocessing, model loading, prediction API.
10. Implement Grad-CAM explainability: Grad-CAM function returning heatmaps and overlayed images.
11. Build GUI skeleton (`app/`): `main_app.py` layout, patient registry sidebar, capture and run buttons.
12. Integrate camera capture/live feed: OpenCV video capture, brightness/contrast controls, stable frame capture.
13. Wire inference into GUI: connect capture -> inference -> results display (species + confidence + Grad-CAM).
14. Implement SQLite logging API: consistent schema (`clinical_records.db`), insert/query/export functions.
15. Add report export functionality: CSV/PDF export of recent logs and single-case reports with images.
16. Unit & integration tests: tests for preprocessing, inference, Grad-CAM, DB operations, and GUI flows.
17. Performance optimization & quantize: profiling, reduce model size (TF Lite/quantization) for faster offline inference.
18. Packaging for offline deployment: build distributable (standalone binary, Docker image, or installer) and offline install docs.  (Docker image + scripts added: `Dockerfile`, `scripts/build_offline_package.sh`, `scripts/install_offline_package.sh`)
19. Hardware calibration utilities: `microscope_settings.json` editor and calibration workflow (scale, brightness, focus presets).
20. Clinical validation & ethics review: prepare study protocol, collect labelled validation set, obtain approvals.
21. Documentation & user manual: full `README.md` update, GUI user guide, maintenance notes, and quickstart.
22. On-site deployment & training: install at MRRH, run acceptance tests, train lab staff, collect feedback.
23. Maintenance plan & monitoring: scheduled retraining, model versioning, backup and data retention policies.

---

Next steps: start with step 2 (repo cleanup) and step 3 (requirements). If you want, I can begin by creating `requirements.txt` and a `TODO` issue board.
