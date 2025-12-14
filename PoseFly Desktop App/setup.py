# setup.py
import sys
from cx_Freeze import setup, Executable

# (Optional) helps if some hook still recurses a bit before exclusions kick in
sys.setrecursionlimit(5000)

base = "Win32GUI" if sys.platform == "win32" else None

build_exe_options = {
    # IMPORTANT: do NOT force a big "packages": [...] list.
    # Let cx_Freeze discover dependencies naturally.

    "include_files": [
        ("machinelearning/models", "machinelearning/models"),
        ("assets", "assets"),
    ],

    # The important part: prevent the tensorflow/tensorboard hook loop
    "excludes": [
        "tensorflow",
        "tensorboard",
        "tensorboardX",
        "keras",

        # Optional exclusions (usually not needed for .pt inference)
        "onnx",
        "onnxruntime",
        "openvino",
        "tensorrt",
        "cv2.gapi",
    ],

    "optimize": 1,
}

setup(
    name="PoseFly",
    version="1.0",
    description="AI-enhanced, camera-based drone-to-drone communication system",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main.py",
            base=base,
            target_name="PoseFly.exe",
            icon="assets/posefly.ico",
        )
    ],
)