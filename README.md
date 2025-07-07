# QUESTDB benchmark of L-PDFT and MC-PDFT using meta-GGA functionals
This work is published in https://arxiv.org/abs/2506.03304

## Python packages version
    Python=3.10.6
    numpy=2.1.3
    pandas=2.2.3
    pyscf=2.9.0

## How to use
Download the previous benchmark data from [text](https://zenodo.org/records/6644169).
After unzip, choose one directory (e.g. Aug1212) and run:
```sh
cp /path/to/raw_data/Aug1212/*.pkl /path/to/metagga/cas_results/
```

### splittask.py
Example usage: 
```sh
python splittask.py cas_results nsplit -w workdir -j job_prefix
```
Running this script generates `<workdir>` directory, and makes `<nsplit>` number of txt files named `<job_prefix>_<i>.txt`, `<i>` ranging from 0 to `<nsplit>`-1.
Each txt files have roughly equal number of data paths to run.

### runandrun.py
Example usage:
MC-PDFT
```sh
cd workdir
python ../runandrun.py job_prefix_i.txt -o ../otfnals.txt -s subdir --molden
```
L-PDFT
```sh
python runandrun.py job_prefix_i.txt -o ../otfnals.txt -s subdir --molden --linear
```
If you do not want to generate molden files, remove the flag "--molden".

## otfnals.txt
You can add functionals you would like to test in `otfnals.txt`. List of supported functionals in PySCF can be found in [text](https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc_funcs.txt).
Note that all functionals except `MC23` must have `t` or `ft` added in front of its name, indicating that translated or fully-translated version will be used for MC-PDFT.