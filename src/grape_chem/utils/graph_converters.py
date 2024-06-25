from rdkit import Chem
# wrapper for the fragmentation
# TODO: make sure it imports properly

def graph_2_frag(smiles, origin_graph, JT_subgraph):
    mol = Chem.MolFromSmiles(smiles)
    frag_graph_list, motif_graph, atom_mask, frag_flag = JT_subgraph.fragmentation(origin_graph, mol)
    return frag_graph_list, motif_graph, atom_mask, frag_flag