# SURVILLENCE TRAFFIC — AI Traffic Surveillance Platform

A local traffic-video analysis application using the bundled IISc UVH-26 vehicle model, ByteTrack-style temporal tracking, lane analysis, calibrated speed gates, incident detection, and a browser dashboard.

## Windows quick start

1. Install **64-bit Python 3.10, 3.11, 3.12, or 3.13** from python.org.
2. Double-click **`START.bat`**.
3. Wait for the first-time dependency installation to finish.
4. Open **http://localhost:5000**.

`START.bat` automatically creates an isolated `.venv`. It prefers 64-bit Python 3.13 → 3.12 → 3.11 → 3.10 and recreates the virtual environment when it was built with an unsupported Python version. Dependency installation is binary-wheel-only so a missing wheel fails clearly instead of trying to compile NumPy/OpenCV with Visual Studio.

## Portable AI runtime

The supported local runtime executes the bundled `traffic_ai_complete_improved/vehicle_traffic.onnx` model through **OpenCV DNN**. This avoids requiring PyTorch/Ultralytics on every laptop and avoids the NumPy source-build problem shown by old environments on Python 3.13.

The `.pt` checkpoint is still bundled as an optional fallback for environments that already have Ultralytics installed. The normal `START.bat` path does not install that heavy dependency chain.

The ONNX graph uses a fixed **640×640** input in the portable OpenCV path. The dashboard's frame stride and source-frame scaling remain the practical performance controls for this runtime.

## Wrong-way detection

The previous build treated **DOWN** as the default direction. That caused legitimate traffic moving upward in the camera image to be reported as wrong-way.

The current build defaults to **AUTO / lane-flow detection**:

- It first observes the full clip.
- It requires stable direction evidence from multiple confirmed vehicles.
- It learns dominant flow per lane instead of assuming every camera faces the same way.
- A single vehicle is not enough to establish its own legal direction.
- Mixed-flow lanes are not automatically treated as a wrong-way violation.
- Manual `UP`, `DOWN`, and `Both Directions` modes remain available.

The analysis report records the detected scene flow and lane-flow decisions so the result can be audited.

## Browser-compatible output video

The rendered result is first created with OpenCV and then converted to **H.264 / yuv420p / fast-start MP4** using the bundled `imageio-ffmpeg` binary when available. This is designed for Chrome/Edge playback on other Windows laptops without requiring a system FFmpeg installation.

## Dependencies

The normal local runtime is intentionally small:

- Flask
- Flask-CORS
- OpenCV (headless build)
- NumPy
- imageio-ffmpeg

NumPy is selected by Python version so Python 3.13+ uses a wheel-based 2.x release rather than trying to build the old 1.26.x source package.

## Data

Runtime databases and uploaded videos are kept in writable runtime storage. Historical uploaded user data from development machines is **not** included in the distributable project zip.

## Optional model-development environment

The `traffic_ai_complete_improved/requirements.txt` file is for model development/training and is not required for the normal local application. Use the root `START.bat` for the supported application workflow.

## Important vehicle-class fix

The bundled UVH-26 ONNX model embeds its 14 classes in the original training order.
The portable detector now uses that exact order instead of an alphabetical list.
This fixes the previous failure where motorcycles/two-wheelers could be decoded as Sedan
and therefore displayed and counted as cars.

## Reliability defaults

AUTO wrong-way detection is conservative: lane flow is learned only from multiple
confirmed tracks with strong agreement. Uncalibrated speed estimates are informational
only and do not create speed-violation incidents.

## Verification

Before distribution, the portable detector is checked against the bundled model's
14-class ONNX index order and the application mapping. Class id 7 is explicitly
`Two-wheeler -> motorcycle`, preventing the previous motorcycle-as-car regression.


## v4 reliability fixes

- Motorcycle/two-wheeler class ID mapping is locked to the bundled UVH-26 ONNX class order.
- Speed violations work in AUTO mode by default: the detector uses the configured meters-per-pixel estimate immediately and switches to known-distance gates when a reference distance is supplied.
- Speed violations require sustained evidence across multiple observations to avoid one-frame spikes.
- Speed-gate crossing detection scans the complete track history, so gates are not missed just because they are far apart.
- Live camera speed alerts use a safe default estimate (0.05 m/px) when a camera has no saved calibration.
- `verify_install.py` and `VERIFY.bat` provide local regression checks for model loading, class mapping, and speed logic.

Speed values are camera-calibrated estimates unless known-distance speed gates are configured. They should be calibrated before enforcement use.
