import torch
import os

import const

def save_xyz_file(path, one_hot, positions, node_mask, names, suffix=''):
    idx2atom = const.IDX2ATOM

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        n_atoms = mask.sum()
        atom_idx = torch.where(mask)[0]

        f = open(os.path.join(path, f'{names[batch_i]}_{suffix}.xyz'), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in atom_idx:
            atom = atoms[atom_i].item()
            atom = idx2atom[atom]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]
            ))
        f.close()


def load_xyz_files(path, suffix=''):
    files = []
    for fname in os.listdir(path):
        if fname.endswith(f'_{suffix}.xyz'):
            files.append(fname)
    files = sorted(files, key=lambda f: -int(f.replace(f'_{suffix}.xyz', '').split('_')[-1]))
    return [os.path.join(path, fname) for fname in files]


