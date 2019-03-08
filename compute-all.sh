dir=$1
gold=$2

for i in {0..9}; do
    file=$dir.${i}
    echo $file
    ./run_wer_zh.sh $gold $file
done

