import os
import argparse
from glob import glob

def splittask(dirname, nsplit, workdir, job_prefix):
    dirname = os.path.abspath(dirname)
    joblist = glob(os.path.join(dirname, '*.pkl'))
    # joblist.sort()
    njob = len(joblist)
    njobpertask = njob // nsplit + 1
    idxlist = list(range(0, njob, njobpertask))
    idxlist.append(njob)
    
    os.makedirs(workdir, exist_ok=True)

    for i in range(nsplit):
        joblist_parse = joblist[idxlist[i]:idxlist[i+1]]
        out_fn = os.path.join(workdir, f'{job_prefix}_{i}.txt')
        with open(out_fn, 'w') as f:
            f.write('\n'.join(joblist_parse))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str, help='Directory containing pkl files')
    parser.add_argument('nsplit', type=int, help='Number of jobs to split into')
    parser.add_argument('--workdir', '-w', type=str, default='.', help='Working directory for output files')
    parser.add_argument('--job_prefix', '-j', type=str, help='Prefix of txt files')
    args = parser.parse_args()

    dirname = args.dirname
    nsplit = args.nsplit
    workdir = args.workdir
    job_prefix = args.job_prefix
    splittask(dirname, nsplit, workdir, job_prefix)
