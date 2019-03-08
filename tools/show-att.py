import sys
if len(sys.argv) == 0:
    print 'python show-att.py src out att'
    sys.exit()

for s1,s2,s3 in zip(open(sys.argv[1]),open(sys.argv[2]),open(sys.argv[3])):
    print s1.strip()
    print s2.strip()
    print s3.strip()
    s1 = s1.split()
    s2 = s2.split()
    s3 = s3.split()
    for w,a in zip(s2,s3):
        print w,s1[int(a)]
    print

