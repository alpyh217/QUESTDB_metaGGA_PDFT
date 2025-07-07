import os, sys
import argparse
import pickle
import logging
import pandas as pd
from runmcpdft import runlpdft, runmcpdft
import traceback

def save_pkl(data, filename):
    with open(filename,"wb") as file:
        pickle.dump(data, file)

def run(task_fn, runfunc, otfnal, savedir='.', sym_tsfm=True, save_molden=False):
    with open(task_fn, 'r') as f:
        tasklist = f.readlines()
    tasklist = [task.strip() for task in tasklist]
    ntask = len(tasklist)
    
    jobname = task_fn.split('.')[0]

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    logging.basicConfig(
        filename=f"{jobname}.log",
        level=logging.DEBUG,
        format="{asctime} - {levelname} - {message}",
        filemode='w',
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"Starting {jobname} with {ntask} tasks")
    if isinstance(otfnal, list):
        logging.info(f'Got {len(otfnal)} functionals: {otfnal}')
    else:
        logging.info(f'Got 1 functional: {otfnal}')

    for i, pkl_fn in enumerate(tasklist):
        if pkl_fn.startswith('!'):
            logging.info(f'{i+1}/{ntask} Skipping: {pkl_fn.strip("!")}')
            continue
        vte_name = os.path.basename(pkl_fn).split('.')[0]
        if save_molden:
            save_molden_fn = os.path.join(savedir, vte_name+'.molden')
        else:
            save_molden_fn = None
        save_pkl_fn = os.path.join(savedir, vte_name+'.pkl')
        try:
            logging.info(f'{i+1}/{ntask} Running: {save_pkl_fn}')
            if isinstance(otfnal, list):
                energies = []
                for ot in otfnal:
                    energies.append(runfunc(pkl_fn, ot, molden_fn=save_molden_fn, sym_tsfm=sym_tsfm))
                    logging.debug(f'{i+1}/{ntask} {ot} done')
                energies = pd.concat(energies, axis=1)
                energies = energies.loc[:, ~energies.columns.duplicated()] # remove duplicate columns, especially 'sym'
            else:
                energies = runfunc(pkl_fn, otfnal, molden_fn=save_molden_fn, sym_tsfm=sym_tsfm)
            save_pkl(energies, save_pkl_fn)
            logging.info(f'{i+1}/{ntask} Successfully saved: {save_pkl_fn}')
        except RuntimeError as e:
            print(traceback.format_exc())
            print(e)
            logging.error(f'{i+1}/{ntask} Failed: {pkl_fn}')
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task_fn', help='Task list file')
    parser.add_argument('--otfnal', '-o', help='on-top functional, str (separated by @) or file path')
    parser.add_argument('--linear', help='run LPDFT', action='store_true', default=False)
    parser.add_argument('--savedir', '-s', help='save directory', default='.')
    parser.add_argument('--sym_tsfm', help="transform symmetry", default=False, action='store_true')
    parser.add_argument('--molden', help="save molden", default=False, action='store_true')
    args = parser.parse_args()
    
    task_fn = args.task_fn
    otfnal = args.otfnal
    if os.path.isfile(otfnal):
        with open(otfnal, 'r') as f:
            otfnal = f.readlines()
            otfnal = [x.strip() for x in otfnal]
    elif '@' in otfnal:
        otfnal = args.otfnal.split('@')
    linear = args.linear
    savedir = args.savedir
    sym_tsfm = args.sym_tsfm
    save_molden = args.molden

    if linear:
        func = runlpdft
    else:
        func = runmcpdft
    run(task_fn, func, otfnal, savedir, sym_tsfm, save_molden)
