# PoseFly — Webcam Inference (AI-enhanced, Camera-based Drone-to-Drone Communication System)

This repository contains a **webcam-based inference demo** inspired by **PoseFly**, a **4-in-1 Optical Camera Communication (OCC)** approach for swarming drones that integrates:
1) **Drone identification** (active optical label via rolling-shutter OOK),
2) **On-site localization / pose parsing** (distance, angle, speed estimation),
3) **Quick-link communication** (LED-to-camera burst data),
4) **Lighting** (LEDs as illumination). :contentReference[oaicite:0]{index=0}

PoseFly uses rolling-shutter cameras and simple LED hardware to support **drone-to-drone interactions** without relying on congested RF-only centralized control. :contentReference[oaicite:1]{index=1}

---

## What this codebase does

In this implementation, we run a **webcam inference pipeline** that can optionally enable/disable:
- 🛸 **Drone Detection**
- 🧭 **Angle Estimation**
- 📏 **Distance Estimation**
- 💡 **LED ID / Optical Label ID**
- 🎞️ **Optional video saving**

The original PoseFly paper reports strong prototype results (e.g., near 100% distance estimation accuracy up to 20 m, identification up to 12 m, angle/speed within 4 m, and ~5 Kbps quick-link throughput on prototypes). :contentReference[oaicite:2]{index=2}

---

##