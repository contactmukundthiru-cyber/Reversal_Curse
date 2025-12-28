#!/bin/bash
# Run the dashboard server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Flask environment
export FLASK_ENV=${FLASK_ENV:-development}
export FLASK_DEBUG=${FLASK_DEBUG:-true}

echo "Starting Reversal Curse Dashboard..."
echo "Access at: http://localhost:5001"
echo "Login: admin / (see DASHBOARD_PASSWORD in .env)"
echo ""

python -m dashboard.app
