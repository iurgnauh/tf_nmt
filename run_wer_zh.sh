#KALDI_PATH=kaldi/src/bin/

#cat $1 | python ${TOOL_PATH}/add_space.py | awk -F"\t" '{ print NR"\t"$0 }' - > $1.2col
#cat $2 | python ${TOOL_PATH}/add_space.py | awk -F"\t" '{ print NR"\t"$0 }' - > $2.2col
#cat $1 | python ./remove_punc.py | awk -F"\t" '{ print NR"\t"$0 }' - > $1.2col
#cat $2 | python ./remove_punc.py | awk -F"\t" '{ print NR"\t"$0 }' - > $2.2col
cat $1 | awk -F"\t" '{ print NR"\t"$0 }' - > $1.2col
cat $2 | awk -F"\t" '{ print NR"\t"$0 }' - > $2.2col
#cat $1 > $1.2col
#cat $2 > $2.2col
#./compute-wer2 --text --mode=present text_transcipt_tn.txt text_ASR_tn.txt align_out.txt
./compute-wer2 --text --mode=present ark:$1.2col ark:$2.2col align_out.txt
#${KALDI_PATH}/compute-wer --text --mode=present ark:$1.2col ark:$2.2col
