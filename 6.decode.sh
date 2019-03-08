#!/usr/bin/env bash
#set -x

##################
### inputs
src=`pwd`/data/tiny/test.zh
ref=`pwd`/data/tiny/test.en

### options
#device=/gpu:0

decoder=batch_decoder
#decoder=decoder

#batch=1
batch=3
#batch=10
###################

trg=$src.trans
log=$trg.log
pid=log/train.pid

if [ -f $pid ]; then
	echo already running, pid=`cat $pid`
else
    if [ "$device" != "" ]; then
        device_option="--using_device=$device"
    fi
	python nmt_runner.py \
        --action_type=$decoder \
        --src_test_file=$src \
        --test_batch_size=$batch \
        $device_option \
        >$trg &# \
	    #2>$log &

	echo $! > $pid
	wait

	if [ -f $ref ]; then
		bleu=`sh tools/evaluate_bleu.sh $src $ref $trg`
		trg_new=$trg.bleu-$bleu
		mv $trg $trg_new
		trg=$trg_new
	else
		ref=""
	fi

	head $src $ref $trg $log
	tail $src $ref $trg $log

	ls -l $src $ref $trg $log
	wc -l $src $ref $trg $log

	echo translation src: $src
	echo translation ref: $ref
	echo translation trg: $trg
	echo translation log: $log
	echo translation bleu: $bleu

	rm $pid
fi