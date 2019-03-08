### Train
1. prepare train.src train.trg dev.src dev.trg in ./data
2. build src and trg vocab with 'python build_vocab.py vocab_num < train_file > vocab_file' or simply check and run ./0.build_vocab.sh
3. modify config.py according to your need
4. ./1.run.sh check and run main traing process, ./2.stop.sh stop training process

### Decode
1. ./5.validate.sh run validate backgroud process while training or finetuning
2. ./6.decode.sh after training decode specific dev file and evaluate

### Multi Dev Support
1. before traing, prepare dev.{src,trg}.{0,1,2...} in ./data, and run ../tools/merge_dev.sh
2. after decode, goto ./data and run ../tools/eval_multi_dev.sh

### Added by Youmu
#### Usage
* Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python nmt_runner.py config.py > log/train.log 2>&1 &
```

* Evaluate

```
CUDA_VISIBLE_DEVICES=5 nohup python eval.py config.py > log/eval.log 2>&1 &
```


