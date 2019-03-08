root=$(dirname $0)

if [ $# -ne 3 ]; then
	echo "Usage: $0 <src> <ref> <tst>" 1>&2
	exit -1
fi

src=$1
ref=$2
tst=$3

test -f $src.sgm || bash $root/plain2sgm.sh src $src > $src.sgm
test -f $ref.sgm || bash $root/plain2sgm.sh ref $ref > $ref.sgm
bash $root/plain2sgm.sh tst $tst > $tst.sgm
perl $root/mteval-v11b.pl -r $ref.sgm -s $src.sgm -t $tst.sgm | grep 'BLEU score' | head -n 1 | cut -d ' ' -f 9
rm $src.sgm $ref.sgm $tst.sgm -f

