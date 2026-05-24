#!/bin/zsh
cd /Users/soham/UPenn/Claude/hier_current
/usr/bin/python3 -u -W ignore run_experiments.py "$@" > /tmp/dgp_run.log 2>&1 &
echo $! > /tmp/dgp_pid.txt
echo "Started PID $(cat /tmp/dgp_pid.txt)"
