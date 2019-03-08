import os, sys, re, numpy as np

batch_size=100

r=re.compile('epoch (\d+) 	 updates (\d+) train cost ([\d\.]+) use time ([\d\.]+)')

tt=[]
epoch_tt={}
for fn in sys.argv[1:]:
	for line in file(fn):
		ms=r.findall(line)
		if ms:
			#print line, r, ms
			epoch, updates, cost, time=ms[0]
			epoch, updates, cost, time=int(epoch), int(updates), float(cost), float(time)
			tt.append(time)
			if epoch not in epoch_tt:
				epoch_tt[epoch]=time
			else:
				epoch_tt[epoch]+=time
			#raw_input()

tt2=[]
for k in sorted(epoch_tt.keys())[:-1]:
	tt2.append(epoch_tt[k])

print 'total-time: %0.2fh' % (np.array(tt).sum() / 3600.)
print 'total-batchs: %d' % len(tt)
print 'mean-batch-time: %0.2fs' % np.array(tt).mean()
print 'min-batch-time: %0.2fs' % np.array(tt).min()
print 'max-batch-time: %0.2fs' % np.array(tt).max()
print 'batch-size: %d' % batch_size
print 'finished-epochs: %d' % len(tt2)
print 'total-epochs: %0.2f' % (np.array(tt).sum() / np.array(tt2).mean())
print 'mean-epoch-time: %0.2fh' % (np.array(tt2).mean() / 3600. )
print 'min-epoch-time: %0.2fh' % (np.array(tt2).min() / 3600. )
print 'max-epoch-time: %0.2fh' % (np.array(tt2).max() / 3600. )
print 'total-sentences-trained: %d' % (len(tt)*batch_size)
print 'total-sentences-in-corpus: %d' % (len(tt)*batch_size / (np.array(tt).sum() / np.array(tt2).mean()))
print 'total-sentences-per-day: %d' % (len(tt)*batch_size / (np.array(tt).sum()/3600./24))
print 'total-batchs-per-day: %.2f' % (len(tt) / (np.array(tt).sum()/3600./24))
print 'total-epochs-per-day: %.2f' % (24 * 3600 / np.array(tt2).mean())
