"""Local regression checks for SURVILLENCE TRAFFIC."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "api"))
import detector

expected = [
    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler",
    "Two-wheeler", "LCV", "Mini-bus", "tempo-traveller", "bicycle", "Van", "Others"
]
for i, name in enumerate(expected):
    assert detector.source_label_from_class_id(i) == name
assert detector.app_label_from_source("Two-wheeler") == "motorcycle"
assert detector.app_label_from_source("Sedan") == "car"
assert detector.app_label_from_source("Three-wheeler") == "auto"

assert detector._speed_violation_confirmed([52, 54, 55, 53, 54], 40, 3, 5)
assert not detector._speed_violation_confirmed([38, 39, 41, 39, 38], 40, 3, 5)

gate_speed, crossings = detector._gate_speed_kmh(
    [(100, 200, 0), (100, 250, 27)], 30, 15, 200, 250
)
assert gate_speed is not None and abs(gate_speed - 60.0) < 0.01
assert crossings["gate_a"] == 0.0 and crossings["gate_b"] == 27.0

model = detector.get_yolo_detector()
assert model is not None

print("PASS: class mapping")
print("PASS: speed violation confirmation")
print("PASS: speed-gate calculation")
print(f"PASS: detector load ({model[0]})")
print("ALL REGRESSION CHECKS PASSED")
