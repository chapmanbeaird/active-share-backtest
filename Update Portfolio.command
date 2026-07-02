#!/bin/bash
# Double-click this file in Finder to build the current portfolio from the
# FactSet export you placed in the data/incoming folder.

# Always run from the folder this script lives in (so file paths and the data
# cache resolve correctly no matter where it is launched from).
cd "$(dirname "$0")" || exit 1

echo "=================================================="
echo " Active Share Portfolio - Quarterly Update"
echo "=================================================="
echo ""

# Use the project's virtual environment if it exists; otherwise create it.
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "First-time setup: creating the Python environment (one time only)..."
    python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt
fi

python3 update_portfolio.py
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "Finished successfully. Your portfolio Excel is in the results/portfolios folder."
elif [ $STATUS -eq 2 ]; then
    echo "Action needed: a few new stocks must be classified (see the message above)."
else
    echo "Something went wrong (see the message above)."
fi

echo ""
echo "You can close this window."
# Keep the Terminal window open so the result stays visible.
read -n 1 -s -r -p "Press any key to close..."
