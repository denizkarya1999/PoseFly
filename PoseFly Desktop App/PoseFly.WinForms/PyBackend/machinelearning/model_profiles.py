"""Checkpoint-specific capture settings, independent of the ML runtime.

Outdoor training record: 2026-03-03, PoseFly research conversation.
All four outdoor feature datasets used 1/6000 s and ISO 1000.
The older ISO 800 records refer to earlier datasets, not these checkpoints.
Drone detection deliberately has no feature-exposure preset.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureModel:
    path: str
    label: str
    iso: int
    shutter_hz: float


FEATURE_MODELS = {
    "angle": FeatureModel("models/OutdoorModels/PoseFly_Outdoor_Angle_Detection_YOLO_v26m_with_Self_Attention_and_C3K2_SA.pt", "Angle", 1000, 6000.0),
    "distance": FeatureModel("models/OutdoorModels/PoseFly_Outdoor_Distance_Detection_YOLO_v26m_with_Self_Attention_and_C3K2_SA.pt", "Distance", 1000, 6000.0),
    "led": FeatureModel("models/OutdoorModels/PoseFly_Outdoor_LED_ID_Detection_YOLO_v26m_with_Self_Attention_and_C3K2_SA.pt", "LED ID", 1000, 6000.0),
    "speed": FeatureModel("models/OutdoorModels/PoseFly_Outdoor_Speed_Detection_YOLO_v26m_with_Self_Attention_and_C3K2_SA.pt", "Speed", 1000, 6000.0),
}
