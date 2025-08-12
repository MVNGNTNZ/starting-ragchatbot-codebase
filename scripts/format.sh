#!/bin/bash

# Code formatting script
echo "🎨 Formatting code..."

# Format with black
echo "Running black..."
uv run black backend/

# Sort imports with isort
echo "Sorting imports with isort..."
uv run isort backend/

echo "✅ Code formatting complete!"