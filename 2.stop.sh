#!/usr/bin/env bash
#set -x

pid=log/train.pid
log=log/train.log

if [ -f $pid ]; then
	echo killing pid=`cat $pid`
	kill -9 `cat $pid` && echo done || echo failed
	rm $pid
else
	echo not running
fi
