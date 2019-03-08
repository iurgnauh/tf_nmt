root=$(dirname $0)
test -f $root/plain2sgm || g++ $root/plain2sgm.cpp -o $root/plain2sgm && $root/plain2sgm $*