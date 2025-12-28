# python_planet_simulation

Procedural planet generation project in Python.

Goal: generate a random planet with different biomes, and allow rotating and zooming it.

Dependencies (install via pip):

- pygame
- numpy
- pillow
- opensimplex

Quick start:

1. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv venv
   ```

   - Windows:

   ```bash
   venv\Scripts\activate
   ```

   - Other systems:

   ```bash
   source venv/bin/activate
   ```

   Then install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run:

   ```bash
   python main.py
   ```

Controls:

- Click and drag (left mouse): rotate the planet
- Mouse wheel: zoom in / out
- R: regenerate a random planet

This is a prototype. Performance can be improved in future iterations.
