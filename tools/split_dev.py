import os, sys

if len(sys.argv)!=3:
	print 'Usage: %s <dev.src> <dev.out>' % sys.argv[0]
	sys.exit()

fsrc, fout=sys.argv[1], sys.argv[2]
with open(fout) as f:
	out=f.readlines()
base=0
for i in xrange(10):
	fp=fsrc+'.'+str(i)
	if not os.path.exists(fp):
		break
	with open(fp) as f:
		n=len(f.readlines())
	with open(fout+'.'+str(i), 'w') as f:
		for j in xrange(n):
			f.write(out[base+j])
	base+=n
