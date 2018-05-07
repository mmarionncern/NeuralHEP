#import pandas as pd
#pd.read_hdf("TTWToLNu_ext2_part2.h5")


import h5py
filename = 'TTWToLNu_ext2_part2.h5'
f = h5py.File(filename, 'r')

#with h5py.File('the_filename', 'r') as f:
#    my_array = f['array_name'][()]


for item in f.attrs.keys():
    print(item + ":", f.attrs[item])

# List all groups
print("Keys: %s" % f.keys())
a_group_key = f.keys()[0]
print a_group_key

# Get the data
data = f[a_group_key]
#print data[0][0]

print "size -> ",len(data)

#nJet:Jet_pt:Jet_eta:Jet_phi:Jet_mass:Jet_btagCSV:Jet_qgl:jetIdx:Jet_corr_JECUp:Jet_corr_JECDown:nLepGood:lepWIdx:lepZIdx:LepGood_pt:LepGood_eta:LepGood_phi:LepGood_pdgId:met_pt:met_phi

#order : (I/F/AI/AF) and alphanumerical ordering inside each category

#for evt in data[0:5]:
#    print "=============== new event =========="
#    print "nJet:",     evt[0]
#    print "nLepGood:", evt[1]
#    print "lepWIdx",   evt[2]
#    print "met_pt",   evt[3]
#    print "met_phi",   evt[4]
#    print "LepGood_pdgId:", evt[5]
#    print "jetIdx:",        evt[6]
#    print "lepZIdx:",       evt[7]
#    print "Jet_btagCSV:",     evt[8]
#    print "Jet_corr_JECDown:", evt[9]
#    print "Jet_corr_JECUp:",     evt[10]
#    print "Jet_eta:",     evt[11]
#    print "Jet_mass:",     evt[12]
#    print "Jet_phi:",     evt[13]
#    print "Jet_pt:",     evt[14]
#    print "Jet_qgl:",     evt[15]
#    print "LepGood_eta:",     evt[16]
#    print "LepGood_phi:",     evt[17]
#    print "LepGood_pt:",     evt[18]
  
