import multiprocessing
import os
import torch
import pickle
import random
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import categorical_edge_match, categorical_node_match
from rdkit import Chem
from rdkit.Chem import AllChem

ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P':8}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P':15}


def sdf2dict(sdf_path, pkl_path, prefix):
    with open(sdf_path) as f:
        lines = f.readlines()
    sdf = {}
    smi_name = None
    for line in tqdm(lines):
        if line.startswith(prefix) and len(line.split('_'))==2 and '-' not in line:
            nodes_id = 0
            smi_name = line.strip().split('_')[1]
            sdf[smi_name] = nx.Graph()
        split_line = line.split()
        if len(split_line) == 16:
            sdf[smi_name].add_node(
                nodes_id, 
                element=split_line[3], 
                positions=[float(_) for _ in split_line[:3]]
            )
            nodes_id += 1
        if len(split_line) == 4:
            sdf[smi_name].add_edge(int(split_line[0])-1, int(split_line[1])-1, type=split_line[2])
    with open(pkl_path,'wb') as f:
        pickle.dump(sdf,f)
    return sdf


def extract_from_G_and_G_linker(G, G_linker, test_ids):
    nm = categorical_node_match("element",['C','N','O','S','F','Cl','Br','I','P'])
    em = categorical_edge_match("type",['1','2','3','4','5'])
    GM = isomorphism.GraphMatcher(G, G_linker,node_match=nm, edge_match=em)
    maps = []
    anchors = []
    tmp_n = 0
    for i in GM.subgraph_isomorphisms_iter():
        if tmp_n<1000:
            tmp_n+=1
            n0 = G.nodes
            n1 = list(i.keys()) # linker
            n1.sort()
            n2 = list(set(n0) - set(n1)) # ligand
            e0 = G.edges
            e1 = G.subgraph(n1).edges
            e2 = G.subgraph(n2).edges
            if len(e1) + len(e2) + 2 == len(e0) and (n1 not in maps):
                maps.append(n1)
                lost_e = list(set(e0) - set(e1)-set(e2))
                anchor = []
                for j in lost_e:
                    if j[0] not in n1: anchor.append(j[0])
                    if j[1] not in n1: anchor.append(j[1])
                anchors.append(set(anchor))
        else:
            break

    if len(maps) > 1:
        with open("map",'a') as f:
            f.write(f"maps>1: {test_ids}\n")
    if len(maps) == 0:
        with open("map",'a') as f:
            f.write(f"maps=0: {test_ids}\n")

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

            tmp = [0.]*len(ATOM2IDX)
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
            tmp = [0.]*len(ATOM2IDX)
            tmp[ATOM2IDX[G.nodes[linker_atom]['element']]]=1.
            one_hot.append(tmp)
            charges.append(CHARGES[G.nodes[linker_atom]['element']])
    
        return {
            'uuid': test_ids,
            'name': test_ids,
            'positions': torch.tensor(positions),
            'one_hot': torch.tensor(one_hot),
            'charges': torch.tensor(charges),
            'anchors': torch.tensor(in_anchors),
            'fragment_mask': torch.tensor(fragment_mask),
            'linker_mask': torch.tensor(linker_mask),
            'num_atoms': n,
        }


if __name__ == "__main__":

    # sdf file to dict
    linker = sdf2dict("linker_noH.sdf",'linker.pkl','linker')
    protacs = sdf2dict("protacs_noH.sdf",'protacs.pkl','protacs')
       
    
    test_sets = []
    for test_ids in tqdm(linker.keys()):
        if test_ids in protacs:
            out = extract_from_G_and_G_linker(protacs[test_ids], linker[test_ids], test_ids)
            if out is not None:
                test_sets.append(out)
    # torch.save(test_sets, 'quasi_test_set.pt')
    print(len(test_sets))

    random.shuffle(test_sets)
    torch.save(test_sets[:-800], "protacs_train.pt")
    torch.save(test_sets[-800:-400], "protacs_val.pt")
    torch.save(test_sets[-400:], "protacs_test.pt")

