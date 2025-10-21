#!/bin/bash
# Quick Test - Run FL for 3 rounds only (faster demo)

echo "🧪 QUICK TEST MODE"
echo "Running 3 rounds (instead of 20) for faster testing..."
echo "================================================================"
echo ""

# Create temporary test version
sed 's/num_rounds.*: 20/num_rounds: 3/g; s/local_epochs.*: 3/local_epochs: 2/g' run_clean_fl.py > /tmp/test_fl.py

# Run it
python /tmp/test_fl.py

# Cleanup
rm /tmp/test_fl.py

echo ""
echo "================================================================"
echo "✅ Quick test complete!"
echo ""
echo "To run full training (20 rounds):"
echo "  python run_clean_fl.py"

