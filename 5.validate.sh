export CUDA_VISIBLE_DEVICES=99
nohup python eval.py >> log/eval.log 2>&1 &
tail -f log/eval.log
