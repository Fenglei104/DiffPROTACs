import os
import sys
from rdkit import Chem
from utils import disable_rdkit_logging
disable_rdkit_logging()

gen_smi_path = sys.argv[1]
n = int(sys.argv[2])

valid_cnt = 0
unique_cnt = 0
total_cnt = 0
recovery_cnt = 0

def is_valid(mol_path):
    try:
        mol_gene = Chem.SDMolSupplier(mol_path)[0]
        if mol_gene is None:
            return False
        smi = Chem.MolToSmiles(mol_gene)
        if '.' in smi:
            return False
        Chem.SanitizeMol(mol_gene, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return False

    return True

gen_smi_dirs = os.listdir(gen_smi_path)
for files in gen_smi_dirs:
    mol_true = Chem.SDMolSupplier(f'{gen_smi_path}/{files}/true_.sdf')[0]
    true_smi = Chem.MolToSmiles(mol_true)
    smi_group = []
    for i in range(n):
        total_cnt += 1
        valid = is_valid(f'{gen_smi_path}/{files}/{i}_.sdf')
        if valid:
            valid_cnt += 1
            mol_gene = Chem.SDMolSupplier(f'{gen_smi_path}/{files}/{i}_.sdf')[0]
            smi_group.append(Chem.MolToSmiles(mol_gene))

    if true_smi in smi_group:
        recovery_cnt += 1
    unique_cnt += len(list(set(smi_group)))

validity = valid_cnt / total_cnt * 100
uniqueness = unique_cnt / total_cnt * 100
recovery = recovery_cnt / len(gen_smi_dirs) *100
print(gen_smi_path)
print(f'Validity: {validity:.2f}%')
print(f'Uniqueness: {uniqueness:.2f}%')
print(f'Recovery: {recovery:.2f}%')
