#!/bin/bash

# Complete code quality pipeline
echo "🚀 Running complete code quality pipeline..."

# Format code
echo "Step 1: Formatting code..."
./scripts/format.sh

# Run linting
echo "Step 2: Running quality checks..."
./scripts/lint.sh

# Check if formatting is up to date
echo "Step 3: Checking if code is properly formatted..."
if ! uv run black --check backend/; then
    echo "❌ Code is not properly formatted. Run ./scripts/format.sh"
    exit 1
fi

if ! uv run isort --check backend/; then
    echo "❌ Imports are not properly sorted. Run ./scripts/format.sh"
    exit 1
fi

echo "✅ All code quality checks passed!"