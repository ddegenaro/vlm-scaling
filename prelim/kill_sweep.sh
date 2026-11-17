# 1. stop the loop so it can't start another run
pkill -f sweep.sh

# 2. stop the currently running benchmark
pkill -f smoke_test.py

# 3. confirm both are gone and the GPU memory is freed
pgrep -af "sweep.sh|smoke_test.py"
nvidia-smi