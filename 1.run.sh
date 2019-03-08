#!/usr/bin/env bash
#set -x

# Usage: 1.run.sh <1=restart/0=continue>
export CUDA_VISIBLE_DEVICES=6,7

pid=log/train.pid
log=log/train.log

if [ -f $pid ]; then
	echo already running, pid=`cat $pid`
else
	today=`date '+%Y%m%d.%H%M%S'`
	for d in model log; do
        if [ "$1" == "1" ]; then
		    test -d $d && ( mkdir -p archive/$today && mv $d archive/$today )
        fi
		mkdir -p $d
	done
	mkdir -p log

	nohup python nmt_runner.py >>log/train.log 2>&1 &
	echo $! > $pid
	tail -f log/train.log
fi
