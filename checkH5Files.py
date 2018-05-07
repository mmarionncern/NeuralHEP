import h5py

datasets=['TTZToLLNuNu.h5',
          'WZTo3LNu_amcatnlo.h5',
          'TTHnobb.h5',
          'tZq_ll.h5',
          'backupSummer16/WZTo3LNu.h5',
          'backupSummer16/WZTo3LNu_ext.h5',
          'backupSummer16/tZq_ll.h5',
          'backupSummer16/TTZToLLNuNu_ext2.h5',
]

for ds in datasets:
    f = h5py.File('dataFiles/'+ds, 'r')
    a_group_key = list(f.keys())[0]
    data = f[a_group_key]
    print ds, ":", len(data)
    
