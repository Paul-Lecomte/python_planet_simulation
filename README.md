# python_planet_simulation

Procedural planet generation project in Python.

This prototype generates a random planet texture (elevation, biomes, towns) and renders it with simple 3D lighting and pre-integrated volumetric clouds.

Features

- Procedural continents, mountains, forests and small towns
- Normal map and ambient occlusion for better 3D shading
- Pre-integrated volumetric-style clouds (CPU generated density map, composited each frame)
- Interactive rotation and zoom

Dependencies (install via pip):

- pygame
- numpy
- pillow
- opensimplex

Quick start:

1. Create a virtual environment and install dependencies:

   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1  # on PowerShell (Windows)
   pip install -r requirements.txt
   ```

2. Run the demo:

   ```powershell
   python main.py
   ```

Controls:

- Click and drag (left mouse): rotate the planet
- Mouse wheel: zoom in / out
- R: regenerate a random planet

Notes and performance:

- The planet texture and cloud density are generated on a worker thread at startup to avoid blocking the UI.
- Cloud generation is moderately expensive (samples along a short vertical column). You can lower the cloud sample count or the texture resolution for faster generation.
- This is a prototype: there is room to optimize by moving cloud composition to the GPU or using more efficient noise implementations.

License:

- MIT (use and modify freely)
