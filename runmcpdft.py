import os, sys
import pickle
import numpy as np
import pandas as pd
from pyscf import scf
from pyscf.tools import molden
import pdftutils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import symm

def load_pkl(filename):
    with open(filename,"rb") as file:
        data = pd.read_pickle(file)
    return data

def runmcpdft(pkl_fn, otfnal, molden_fn=None, sym_tsfm=True):
    pkl_fn = os.path.abspath(pkl_fn)
    vte_name = os.path.basename(pkl_fn).split(".")[0]

    print(f"\nTesting MC-PDFT {vte_name} / {otfnal}")
    data = load_pkl(pkl_fn)
    mol = pdftutils.make_molecule(data)
    mo_coeff, ci, nsa, ncas, nelecas, syms = pdftutils.unpack_casdata(data)
    if type(nelecas) == tuple:
        nelecas = int(np.sum(nelecas))

    mol = symm.reduce_point_group(mol)
    if sym_tsfm:
        mol, syms = symm.reduce_symmetries(mol, syms)
    print("Geometry:")
    for atom in mol.atom:
        print(atom)
    print(f"Point group: {mol.topgroup}, converted into {mol.groupname}")
    print("Symmetry:",syms)
    
    print(f"Number of basis: {mol.nao}")
    print(f"Active Space:    ({np.sum(nelecas)},{ncas})")

    print("Setting up ROHF...")
    mf = scf.ROHF(mol)
    mf.mo_coeff = data["mo_coeff"]

    import time
    time_start = time.perf_counter()

    res = {otfnal:[], 'sym':[]}
    for i in range(len(ci)):
        spin = int(syms[i][0]) - 1 # 2S
        nalpha = (nelecas + spin)//2
        nbeta = nelecas - nalpha
        nelecas_tpl = (nalpha,nbeta)

        print(f"State {i+1} : {syms[i]}")
        print("Nactel:",nelecas_tpl)
        print("Running MC-PDFT...")
        mc = pdftutils.make_mcpdft(mf, ncas, nelecas_tpl, syms[i], casorbs=mo_coeff, ci=ci[i],
                                do_scf=False, otfnal=otfnal)
        e_tot, e_ot = mc.energy_tot()

        if i == 0 and molden_fn is not None:
            molden.from_mcscf(mc, molden_fn, cas_natorb=True)

        res[otfnal].append(e_tot)
        res['sym'].append(syms[i])

    res = pd.DataFrame(res)
    time_finish = time.perf_counter()
    time_elapsed = time_finish-time_start
    print(f"Total time elapsed: {np.round(time_elapsed,2)} seconds")
    return res

def runlpdft(pkl_fn, otfnal, molden_fn=None, sym_tsfm=True):
    pkl_fn = os.path.abspath(pkl_fn)
    vte_name = os.path.basename(pkl_fn).split(".")[0]

    print(f"\nTesting L-PDFT {vte_name} / {otfnal}")
    data = load_pkl(pkl_fn)
    mol = pdftutils.make_molecule(data)
    mo_coeff, ci, nsa, ncas, nelecas, syms = pdftutils.unpack_casdata(data)
    if type(nelecas) == tuple:
        nelecas = int(np.sum(nelecas))

    mol = symm.reduce_point_group(mol)    
    if sym_tsfm:
        mol, syms = symm.reduce_symmetries(mol,syms)
    
    print("Geometry:")
    for atom in mol.atom:
        print(atom)
    print(f"Point group: {mol.topgroup}, converted into {mol.groupname}")
    print("Symmetry:",syms)

    print(f"Number of basis: {mol.nao}")
    print(f"Active Space:    ({np.sum(nelecas)},{ncas})")

    print("Setting up ROHF...")
    mf = scf.ROHF(mol)
    mf.mo_coeff = mo_coeff

    import time
    time_start = time.perf_counter()
    print("Running L-PDFT...")
    mc = pdftutils.make_lpdft(mf, ncas, nelecas, syms, casorbs=mo_coeff, ci=ci,
                              do_scf=False, otfnal=otfnal, verbose=4)
    mc.kernel()

    if molden_fn is not None:      
        molden.from_mcscf(mc, molden_fn, cas_natorb=True)

    res = pd.DataFrame({'CASSCF':mc.e_mcscf, f'L-{otfnal}':mc.e_states, 'sym':syms})
    if otfnal == "tPBE":
        res['L-tPBE0'] = mc.e_mcscf * 0.25 + mc.e_states * 0.75
    time_finish = time.perf_counter()
    time_elapsed = time_finish-time_start
    print(f"Total time elapsed: {np.round(time_elapsed,2)} seconds")
    return res
