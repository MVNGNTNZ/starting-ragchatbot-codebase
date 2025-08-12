#!/bin/bash

# Code linting script
echo "🔍 Running code quality checks..."

# Run flake8 linting
echo "Running flake8..."
uv run flake8 backend/

# Run mypy type checking
echo "Running mypy..."
uv run mypy backend/

echo "✅ Code quality checks complete!"