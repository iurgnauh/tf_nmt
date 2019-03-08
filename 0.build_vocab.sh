set -x

# uncomment below to regenerate vocab
#rm -rf vocab

mkdir -p vocab

# assume that src and trg has same vocab size
vocab_size=50000

for lang in zh en; do
	test -f vocab/vocab.$lang || python tools/build-vocab.py $vocab_size < data/train.$lang > vocab/vocab.$lang &
done
wait

head vocab/*.txt
ls -l vocab/
wc -l vocab/*

