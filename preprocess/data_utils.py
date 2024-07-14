import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import categorical_edge_match, categorical_node_match
from rdkit import Chem
from rdkit.Chem import AllChem


def sdf2nx(sdf_path):
    with open(sdf_path) as f:
        sdf = f.readlines()
    G = nx.Graph()
    nodes_id = 0
    for lines in range(len(sdf)):
        tmp = sdf[lines].split()
        if len(tmp) == 16:
            G.add_node(
                nodes_id, 
                element=tmp[3], 
                positions=[float(_) for _ in tmp[:3]]
            )
            nodes_id += 1
        if len(tmp) == 4:
            G.add_edge(int(tmp[0])-1, int(tmp[1])-1, type=tmp[2])
    return G


def get_map_ids_from_nx(G, G_linker):
    nm = categorical_node_match("element",['C','N','O','S','F','Cl','Br','I'])
    em = categorical_edge_match("type",['1','2','3','4','5'])
    GM = isomorphism.GraphMatcher(G, G_linker,node_match=nm, edge_match=em)
    maps = []
    anchors = []
    for i in GM.subgraph_isomorphisms_iter():
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
    return maps, anchors

def smi2sdffile(smi, sdf_path):
    m1 = Chem.MolFromSmiles(smi)
    Chem.Kekulize(m1)
    m2 = Chem.AddHs(m1)
    tmp = AllChem.EmbedMolecule(m2, useRandomCoords=True)
    if tmp<0:
        print(f'{sdf_path} failed')
    else:
        AllChem.MMFFOptimizeMolecule(m2) 
        m3 = Chem.RemoveHs(m2)
        w =  Chem.SDWriter(sdf_path)
        w.SetKekulize(False)
        w.write(m3)
        w.close()
    

def draw_linker(G, maps_ids,save_path):
    plt.figure(figsize=(64,64))
    n0 = G.nodes
    n1 = maps_ids  # linker
    n2 = list(set(n0) - set(n1)) # ligand
    e0 = G.edges
    e1 = G.subgraph(n1).edges
    e2 = G.subgraph(n2).edges
    pos = nx.kamada_kawai_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=n1, node_color="tab:red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=n2, node_color="tab:blue", **options)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=e1, width=8, alpha=0.5, edge_color="tab:red")
    nx.draw_networkx_edges(G, pos, edgelist=e2, width=8, alpha=0.5, edge_color="tab:blue")
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    plt.savefig(save_path+'.png')
