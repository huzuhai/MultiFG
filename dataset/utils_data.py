import time
import pubchempy
import numpy as np

from pubchempy import get_compounds
from rdkit import Chem
from rdkit.Chem import AllChem

def search_from_Pubchem(name):
    time.sleep(2)
    if name == None:
        pass
    else:
        try:
            pubresu = get_compounds(name, 'name')[0]
        except:
            # row['activesubstancename'] = drug
            smiles=None
        else:
            # row['activesubstancename'] = drug
            if pubresu.isomeric_smiles is not None:
                smiles = pubresu.isomeric_smiles
            elif pubresu.canonical_smiles is not None:
                smiles = pubresu.canonical_smiles
            else:
                smiles=None
        return smiles
    
def unify_smiles(smiles):
    # smile -> mol
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = None
    if mol is None:
        return None  #return None if fali
    # standardize SMILE
    standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return standardized_smiles

def getcompund(smiles,drugname):
    try:
        compound = pubchempy.get_compounds(drugname,'name')[0]
    except:
        compound = pubchempy.get_compounds(smiles,'smiles')[0]
    pubchem = compound.cactvs_fingerprint
    pubchem = np.array([int(i) for i in pubchem])
    return list(pubchem)

def fingerpint(smiles, drugname):
    #Macc
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Macc = AllChem.GetMACCSKeysFingerprint(mol)
    Macc = np.array(Macc)
    
    #Morgan
    Morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    Morgan = np.array(Morgan)
    
    #rdkit(Topological)
    #rdkit similary with daylight
    Rtoplo = Chem.RDKFingerprint(mol)
    Rtoplo = np.array(Rtoplo)
    
    # #pharmacophore
    # AllChem.EmbedMolecule(mol)
    # factory = Gobbi_Pharm2D.factory
    # #calc 3d p4 fp
    # phar = Generate.Gen2DFingerprint(mol, factory, dMat = Chem.Get3DDistanceMatrix(mol))
    # phar = [int(i) for i in phar.ToBitString()]
    # phar = np.array(phar)
    return list(Macc), list(Morgan), list(Rtoplo)

def phamfp(smiles):
    #Macc
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    #pharmacophore
    phar = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
    return list(phar)
