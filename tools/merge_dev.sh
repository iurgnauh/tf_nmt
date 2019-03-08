set -x
rm -f dev.en dev.zh
for i in $(seq 0 6); do
	cat dev.en.$i >> dev.en
	cat dev.zh.$i >> dev.zh
done
wc -l dev.en* dev.zh*
