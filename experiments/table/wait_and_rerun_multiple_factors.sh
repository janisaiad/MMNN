#!/bin/bash
# Script to wait for current training to finish, then rerun with multiple factors

echo "⏳ Waiting for current training processes to finish..."
while ps aux | grep -v grep | grep -q "tune_lr_decay_L2.py"; do
    sleep 60  # Check every minute
    echo "   Still running... ($(date))"
done

echo ""
echo "✅ All training processes finished!"
echo "🚀 Relaunching with multiple factors [1, 2, 3, 4, 5]..."
echo ""

cd /Data/janis.aiad/MMNN/experiments/table
python3 tune_lr_decay_L2.py > tune_lr_multiple_factors_all.log 2>&1 &

echo "✅ New training started with factors [1, 2, 3, 4, 5]"
echo "   Log file: tune_lr_multiple_factors_all.log"
echo "   Total configs: 5 factors × 5 ranks × 1 lr_config = 25 configurations"
echo ""
echo "📊 Monitoring progress..."
sleep 5
tail -20 tune_lr_multiple_factors_all.log
