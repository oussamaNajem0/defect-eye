# defect-eye: AI-Driven Software Defect Predictor

`defect-eye` is an automated Software Quality & Defect Prediction tool that parses source code, computes static complexity metrics and git churn indicators, and evaluates defect risk using machine learning models and SHAP explainability.

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/username/defect-eye.git
cd defect-eye

# Install package and dependencies
pip install -e .[dev]
```

### 2. Code Quality & Testing
```bash
# Run linter checks
ruff check .

# Code formatting check
black --check .

# Execute unit tests with coverage
pytest --cov=src tests/
```

## Project Structure
```
defect-eye/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── config.yaml
├── src/
│   └── defect_eye/
│       ├── __init__.py
│       ├── cli.py
│       ├── data/
│       │   └── __init__.py
│       ├── extractor/
│       │   └── __init__.py
│       └── models/
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_sanity.py
├── .gitignore
├── pyproject.toml
└── README.md
```