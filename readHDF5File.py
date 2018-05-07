import h5py

f = h5py.File("dataFiles/Fall17_TTZvstZq.h5", 'r')
weightkey = list(f.keys())[2]
weights = f[weightkey]

for key in weights:
    print key
    for i in weights[key]:
        print "->",i

print len(f['data']), len(f['data']) // 2000
for i,k in enumerate(f['data'][0]):
    print i, "->",k
