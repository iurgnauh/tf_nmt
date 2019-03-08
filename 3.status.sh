#!/usr/bin/env bash
#show cpu processes
ps aux | grep python | grep runner

#show gpu processes
nvidia-smi

#show training status
tail log/train.log

#show training statasitics index
#python tools/status.py < log/train.log

#show bleu history
cat log/eval.log | grep -n1 BLEU
