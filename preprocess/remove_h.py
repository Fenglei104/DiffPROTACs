from rdkit import Chem

mols =  Chem.SDMolSupplier("linker.sdf")
for mol in mols:
    Chem.RemoveHs(mol)

with Chem.SDWriter("linker_noH.sdf") as w:
    for mol in mols:
        w.SetKekulize(False)
        w.write(mol)


mols =  Chem.SDMolSupplier("protacs.sdf")
for mol in mols:
    Chem.RemoveHs(mol)

with Chem.SDWriter("protacs_noH.sdf") as w:
    for mol in mols:
        w.SetKekulize(False)
        w.write(mol)
