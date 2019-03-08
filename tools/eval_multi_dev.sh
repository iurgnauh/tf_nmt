set -x
python $(dirname $0)/split_dev.py dev.en dev.out
for i in $(seq 0 6); do
	ls -l dev.*.$i
	wc -l dev.*.$i
	sh $HOME/tools/wmt/validate.sh dev.en.$i dev.zh.$i dev.out.$i
done
