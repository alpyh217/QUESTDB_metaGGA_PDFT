import os
import pandas as pd
import numpy as np
import pickle
from pyscf import gto, scf, mcscf, mcpdft, fci, mrpt
from pyscf.gto.basis import parse_gaussian
from pyscf.mcscf.addons import state_average_mix

SS_TO_SPINMULT = {
    0   : 1,
    0.75: 2,
    2   : 3,
    3.75: 4
}

def unique_atoms(geom):
    return list({g[0] for g in geom})

def make_molecule(data):
    mol = gto.Mole()
    mol.atom = data["metadata"]["geom"]
    mol.basis = data["metadata"]["basis"]
    mol.charge = data["metadata"]["charge"]
    mol.spin = data["metadata"]["hf_spin"]
    mol.symmetry = True
    mol.unit = data["metadata"]["unit"]
    mol.build()
    return mol

def make_fcisolvers(mf, syms, ci=None):
    twosplus1_to_ss = {int(2*i/2+1):round(i/2*(i/2+1),2) for i in range(0,10)}
    unique_syms, idx = np.unique(syms, return_index=True)
    unique_syms = unique_syms[idx] # Prevent the symmetries be switched the order
    
    solvers = []
    for i,sym in enumerate(unique_syms):
        twosplus1 = int(sym[0])
        twos = twosplus1 - 1
        ss = twosplus1_to_ss[twosplus1]
        wfnsym = sym[1:]

        is_singlet = (ss == 0)
        #Make solver
        solver = fci.solver(mf.mol, singlet=is_singlet, symm=True)
        solver.spin = twos
        solver = fci.addons.fix_spin_(solver,ss=ss)
        solver.wfnsym = wfnsym
        solver.nroots = syms.count(sym)
        if ci is not None:
            if len(syms) == 1:
                solver.ci = ci
            else:
                solver.ci = ci[i]
        solvers += [solver]
        
    return solvers

def unpack_casdata(data):
    mo_coeff = data['mo_coeff']
    ci = data['ci']
    nsa = len(ci)
    ncas = data["metadata"]["ncas"] 
    nelecas = data["metadata"]["nelecas"]
    ss = np.round(data["energies"]["ss"],2)
    syms = data['energies']['sym']
    twosplus1 = [SS_TO_SPINMULT[s] for s in ss]
    syms = [f'{s}{sym}' for s,sym in zip(twosplus1,syms)]
    
    return mo_coeff, ci, nsa, ncas, nelecas, syms

def make_lpdft(mf, ncas, nelecas, syms, casorbs=None, ci=None, do_scf=False, otfnal="tPBE", verbose=4, grids_level=9):
    if do_scf:
        mc = mcpdft.CASSCF(mf, otfnal, ncas, nelecas, natorb=True, grids_level=grids_level)
        mc.max_cycle = 500
    else:
        if casorbs is None:
            raise ValueError("Need to provide mo_coeff for mcpdft.CASCI")
        mc = mcpdft.CASCI(mf, otfnal, ncas, nelecas, natorb=True, grids_level=grids_level)
    solvers = make_fcisolvers(mf, syms, ci=ci)
    
    nstates = np.sum([s.nroots for s in solvers])
    weights = np.ones(nstates)/nstates
    
    mc = mc.multi_state_mix(solvers, weights, "lin")

    if casorbs is not None:
        mc.mo_coeff = casorbs
    if ci is not None:
        assert nstates == len(ci)
        mc.ci = ci
    mc.verbose = verbose
    mc.chkfile = None
    return mc

def make_mcpdft(mf, ncas, nelecas, syms, casorbs=None, ci=None, do_scf=False, otfnal="tPBE", verbose=4, grids_level=9):
    if do_scf:
        mc = mcpdft.CASSCF(mf, otfnal, ncas, nelecas, natorb=True, grids_level=grids_level)
        mc.max_cycle = 500
    else:
        if casorbs is None:
            raise ValueError("Need to provide mo_coeff for mcpdft.CASCI")
        mc = mcpdft.CASCI(mf, otfnal, ncas, nelecas, natorb=True, grids_level=grids_level)
    solvers = make_fcisolvers(mf, syms, ci=ci)
    
    nstates = np.sum([s.nroots for s in solvers])
    weights = np.ones(nstates)/nstates
    
    if nstates == 1:
        mc.fcisolver = solvers[0]
    else:
        mc = state_average_mix(mc, solvers, weights)

    if casorbs is not None:
        mc.mo_coeff = casorbs
    if ci is not None:
        mc.ci = ci
    mc.verbose = verbose
    mc.chkfile = None
    return mc

def virt_transform(mf,l):    
    f_ao = mf.get_fock()
    k_ao = mf.get_k()
    
    if isinstance(mf, scf.uhf.UHF):
        orbs = mf.mo_coeff[0]
        occ = mf.mo_occ.sum(axis=0) #summed occupation
        f_ao = np.sum(f_ao,axis=0)/2 #averaged fock
        k_ao = np.sum(k_ao,axis=0) #summed exchange
    elif isinstance(mf, scf.rohf.ROHF):
        orbs = mf.mo_coeff
        occ = mf.mo_occ.copy()
        k_ao = np.sum(k_ao,axis=0) #summed exchange
    else:
        orbs = mf.mo_coeff
        occ = mf.mo_occ.copy()
        
    def rot(mao,c):
        mmo = np.linalg.multi_dot([c.T,mao,c])
        evals,evecs = np.linalg.eigh(mmo)
        return np.dot(c,evecs)
    
    mat = l*k_ao-f_ao
    
    docc = orbs[:,occ != 0]
    virt = orbs[:,occ == 0]
    virt = rot(mat,virt)
    mf.mo_coeff = np.hstack([docc,virt])
    return mf

def get_gs_es_idx(df):
    df["tpbe0"] = 0.75*df["tpbe"] + 0.25*df["mcscf"]
    df = df.reset_index().drop("index",axis="columns")
    gs_sym = df.iloc[0]["sym"]
    es_sym = df.iloc[-1]["sym"]
    if gs_sym == es_sym:
        df = df.sort_values(by="tpbe0")
        gs_row = df.iloc[0]
        es_row = df.iloc[-1]
    else:
        gs_row = df.iloc[0]
        esdf = df.iloc[1:]
        esdf = esdf.sort_values(by="tpbe0")
        es_row = df.iloc[-1]

    return gs_row.name,es_row.name
    
def calc_ex(df):
    df = df.reset_index().drop("index",axis="columns")
    gs_idx, es_idx = get_gs_es_idx(df)
    gs_row = df.loc[gs_idx].drop(["sym","dw","tdm"])
    es_row = df.loc[es_idx].drop(["sym","dw","tdm"])
    
    ser = es_row - gs_row
    ser = ser.astype("float")
    hartree_to_ev = 27.2114
    ser *= hartree_to_ev
    return ser

def calc_avg(df):
    df = df.reset_index().drop("index",axis="columns")
    gs_idx, es_idx = get_gs_es_idx(df)
    gs_row = df.loc[gs_idx].drop(["sym","dw","tdm"])
    es_row = df.loc[es_idx].drop(["sym","dw","tdm"])

    ser = (es_row + gs_row)/2
    return ser["tpbe"]

def make_tdms(mc,df,dmrg=False):
    fcisolver_spins = []
    for fcisolver in mc.fcisolver.fcisolvers:
        fcisolver_spins += [fcisolver.spin]*fcisolver.nroots
    
    charges = mc.mol.atom_charges()
    coords = mc.mol.atom_coords()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mc.mol.set_common_orig_(nuc_charge_center)
    dip_ints = mc.mol.intor('cint1e_r_sph', comp=3)

    def makedip(idx,dmrg=False):
        assert(idx != gs_idx)
        #Assumes first solver is the GS
        if not dmrg:
            t_dm1 = mc.fcisolver.fcisolvers[0].trans_rdm1(mc.ci[gs_idx], mc.ci[idx], mc.ncas, mc.nelecas)
            orbcas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
            t_dm1_ao = np.linalg.multi_dot([orbcas, t_dm1, orbcas.T])
            tdm = np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
        else:
            print("Not implemented!")
            assert(False)

        tdm = np.sqrt(np.dot(tdm,tdm))
        return tdm

    gs_idx, es_idx = get_gs_es_idx(df)
    gs_spin = fcisolver_spins[gs_idx]
    tdms = []
    for idx in range(df.shape[0]):
        es_spin = fcisolver_spins[idx]
        if gs_spin != es_spin:
            tdms += [None]
        else:
            if idx != gs_idx:
                tdms += [makedip(idx,dmrg=dmrg)]
            else:
                tdms += [None]
    df["tdm"] = tdms

    return df

def run_sanevpt2(mc):
    res = []
    twos_list = []
    for fcisolver in mc.fcisolver.fcisolvers:
        twos_list += [fcisolver.spin]*fcisolver.nroots
    for i in range(len(mc.ci)):
        nalpha, nbeta = mc.nelecas
        twos = twos_list[i]
        while nalpha - nbeta != twos:
            nalpha += 1
            nbeta -= 1
        nelecas = (nalpha,nbeta)
        mc2 = mcscf.CASCI(mc._scf,mc.ncas,nelecas)
        mc2.mo_coeff = mc.mo_coeff
        mc2.ci = mc.ci[i]
        res += [mrpt.NEVPT(mc2).kernel()]
    return np.array(res)
