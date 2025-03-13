# Centrifugal Pump
Centrifugal Pump is a set of tools to help engineers to asses the condition of an API 610 pump during its Performance and Mechanical Running Test trials.

# 🚀Installation and Packaging Guide

This guide explains how to package, distribute, and install the ```centrifugal-pump``` Python library.

## Installing the Package

### ✅ For regular users

To install the package with its core dependencies:

```
git clone https://github.com/eu-cristofer/centrifugal-pump.git
cd centrifugal-pump
pip install .
``` 

### 🛠 For Developers

If you plan to modify or contribute to the project, install it in editable mode with development dependencies:

```
git clone https://github.com/eu-cristofer/centrifugal-pump.git
cd centrifugal-pump
pip install .[dev]
```
This installs the package in editable mode (`-e`) along with all development tools.

## Building the package

Ensure you have the necessary build tools installed:

```pip install build```

Then, generate the distribution files:

```python -m build```

This creates the dist/ folder containing:

- A source distribution (.tar.gz)
- A wheel distribution (.whl)

# TODO

1. Add viscosity correction
2. Add testing
3. Check style guidelines
4. Run type checking
