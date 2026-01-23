#!/bin/bash
# we check the status of the comprehensive training run

cd /Data/janis.aiad/MMNN/experiments/table

echo "=== Training Status ==="
echo ""

# we check if process is running
if pgrep -f train_1d_comprehensive.py > /dev/null; then
    echo "✓ Training process is running"
    PID=$(pgrep -f train_1d_comprehensive.py | head -1)
    echo "  PID: $PID"
    ps -p $PID -o etime,pcpu,pmem,cmd --no-headers | awk '{print "  Runtime: " $1 ", CPU: " $2 "%, Memory: " $3 "%"}'
else
    echo "✗ Training process is not running"
fi

echo ""
echo "=== Progress ==="
if [ -f train_1d_comprehensive.log ]; then
    echo "Last 10 lines of log:"
    tail -10 train_1d_comprehensive.log
    echo ""
    echo "Configurations completed:"
    grep -c "Config.*completed successfully" train_1d_comprehensive.log 2>/dev/null || echo "0"
    echo ""
    echo "Current configuration:"
    tail -30 train_1d_comprehensive.log | grep -E "(CONFIG|Training:|epoch)" | tail -5
fi

echo ""
echo "=== Results Directory ==="
if [ -d results_1d_comprehensive ]; then
    echo "Results directory exists"
    echo "Number of completed runs:"
    ls -d results_1d_comprehensive/*/ 2>/dev/null | wc -l
    echo ""
    echo "Recent results:"
    ls -lt results_1d_comprehensive/*/results.json 2>/dev/null | head -5 | awk '{print "  " $9}'
fi
