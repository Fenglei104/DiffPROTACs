import multiprocessing
import os
import torch
import pandas as pd
from tqdm import tqdm 
import random

from const import NUMBER_OF_ATOM_TYPES, ATOM2IDX, IDX2ATOM, CHARGES
from data_utils import smi2sdffile, sdf2nx, get_map_ids_from_nx


datas = pd.read_csv('protacDB_smiles.csv')
def smi2sdf(i):
    if datas['linker_canonical'][i] == "None":
        print(datas["id_protac"][i])
        return
    smi2sdffile(datas['linker_canonical'][i],f'data/{datas["id_protac"][i]}_linker.sdf')
    smi2sdffile(datas['smiles_canonical'][i],f'data/{datas["id_protac"][i]}_protac.sdf')

if __name__ == "__main__":
    # generate sdf files

    # 3270 -> 3243
    pool = multiprocessing.Pool(30)
    pool.map(smi2sdf, range(3270))
    pool.close()
    pool.join()

    # generate files
    ids = list(set([_.split('_')[0] for _ in os.listdir('data')]))
    train_sets = []
    for id_i in tqdm(ids):
        G = sdf2nx(f'data/{id_i}_protac.sdf')
        G_linker = sdf2nx(f'data/{id_i}_linker.sdf')
        maps, anchors = get_map_ids_from_nx(G, G_linker)
        if len(maps) ==1:
            n = len(G.nodes)
            n0 = G.nodes
            n1 = maps[0] # linker
            n2 = list(set(n0) - set(n1)) # ligand
            positions = []
            one_hot = [] 
            charges = []
            in_anchors = []
            fragment_mask = []
            linker_mask = []

            for ligand_atom in n2:
                positions.append(G.nodes[ligand_atom]['positions'])
                fragment_mask.append(1.)
                linker_mask.append(0.)

                tmp = [0.]*NUMBER_OF_ATOM_TYPES
                tmp[ATOM2IDX[G.nodes[ligand_atom]['element']]]=1.
                one_hot.append(tmp)
                charges.append(CHARGES[G.nodes[ligand_atom]['element']])
                if ligand_atom in anchors[0]:
                    in_anchors.append(1.)
                else:
                    in_anchors.append(0.)
            
            for linker_atom in n1:
                positions.append(G.nodes[linker_atom]['positions'])
                fragment_mask.append(0.)
                linker_mask.append(1.)

                tmp = [0.]*NUMBER_OF_ATOM_TYPES
                tmp[ATOM2IDX[G.nodes[linker_atom]['element']]]=1.
                one_hot.append(tmp)
                charges.append(CHARGES[G.nodes[linker_atom]['element']])

            train_sets.append({
                'uuid': id_i,
                'name': datas['smiles_canonical'][int(id_i)-1],
                'positions': torch.tensor(positions),
                'one_hot': torch.tensor(one_hot),
                'charges': torch.tensor(charges),
                'anchors': torch.tensor(in_anchors),
                'fragment_mask': torch.tensor(fragment_mask),
                'linker_mask': torch.tensor(linker_mask),
                'num_atoms': n,
            })


    random.shuffle(train_sets)
    train_data = train_sets[-800:]
    val_data = train_sets[-800: -400]
    test_data = train_sets[-400:]
    torch.save(train_data, 'protac_train.pt')
    torch.save(val_data, 'protac_val.pt')
    torch.save(test_data, 'protac_test.pt')
