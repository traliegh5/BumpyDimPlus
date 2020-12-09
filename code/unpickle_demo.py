import pickle
fileName = "D://Brown//Senior//CSCI_1470//sampl//neutrMosh//neutrSMPL_CMU//01//01_01.pkl"
with open(fileName, 'rb') as f:
    dd = pickle.load(f, encoding="latin-1") 
print(dd['trans'].shape)
print(dd['poses'].shape)
print(dd['betas'].shape)