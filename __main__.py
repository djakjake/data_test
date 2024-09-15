import numpy as np
import pandas as pd
import h5py as h5
import tables as tb

import argparse
import itertools
import matplotlib.pyplot as plt
import os
import shutil

from datetime import datetime

# =============================================================================
# run options
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('-col','--columns',
                    nargs = "*",
                    default = np.array([1e4], dtype=int).tolist(),
                    type = int,
                    help = "the number of column(s) to use")

parser.add_argument('-row', '--rows',
                    nargs = "*",
                    default = np.arange(1e2, 1e4+1e2, 1e2, dtype=int).tolist(),
                    type = int,
                    help = "the number of row(s) to use")

parser.add_argument('-rep', '--replicants',
                    default = 1,
                    type = int,
                    help = "the number of times to repeat each experiment")

opts = parser.parse_args()

# =============================================================================
# global
# =============================================================================

BYTE_TO_MB = 1 / np.square(1024).item()

class test:
    operations = ['read','write']
    colors = {'hdf5_h5': '#ff0000',
              'hdf5_tb': '#00ff00',
              'hdf5_pd': '#0000ff',
              }
    methods = list(colors.keys())
    shapes = list(itertools.product(opts.rows,opts.columns))
    replicants = range(opts.replicants)
    columns = ['size_read',
               'size_write_MB',
               'test_time'
               ]
    #data = None
    names = dict(operation = operations,
                 method = methods,
                 shape = shapes,
                 replicant = replicants,
                 )
    idx = pd.MultiIndex.from_product(names.values(), names=names.keys())
    data = pd.DataFrame(index=idx, columns=columns)
    data.sort_index(inplace=True)

# =============================================================================
# read functions
# =============================================================================

def read_hdf5_h5(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_h5.hdf5"
    print(f"\t\t\t[read_hdf5_h5] {path}")

    t0 = datetime.now()
    with h5.File(path,'r') as f:
        X_ = f['X']
        size = X_.size
    tf = datetime.now()

    size_read = os.path.getsize(path)
    test_time = get_test_time(t0,tf)
    add_test_data('read','hdf5_h5',size_read,test_time,**kwargs)

def read_hdf5_tb(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_tb.hdf5"
    print(f"\t\t\t[read_hdf5_tb] {path}")

    t0 = datetime.now()
    with tb.open_file(path, mode='r') as f:
        X_ = f.root.X.read()
    tf = datetime.now()

    size_read = os.path.getsize(path)
    test_time = get_test_time(t0,tf)
    add_test_data('read','hdf5_tb',size_read,test_time,**kwargs)

def read_hdf5_pd(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_pd.hdf5"
    print(f"\t\t\t[read_hdf5_pd] {path}")

    t0 = datetime.now()
    X_ = pd.read_hdf(path)
    tf = datetime.now()

    size_read = os.path.getsize(path)
    test_time = get_test_time(t0,tf)
    add_test_data('read','hdf5_pd',size_read,test_time,**kwargs)

# =============================================================================
# write functions
# =============================================================================

def write_hdf5_h5(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_h5.hdf5"
    print(f"\t\t\t[write_hdf5_h5] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    # TODO: support parallel hdf5?
    # TODO: support virtual datasets?
    with h5.File(path,'w') as f:
        X_ = f.create_dataset('X', data=X)
    tf = datetime.now()

    size_read = os.path.getsize(path)
    test_time = get_test_time(t0,tf)
    add_test_data('write','hdf5_h5',size_read,test_time,**kwargs)

def write_hdf5_tb(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_tb.hdf5"
    print(f"\t\t\t[write_hdf5_tb] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    with tb.open_file(path,'w') as f:
        ds = f.create_array(f.root, 'X', X)
    tf = datetime.now()

    size_read = os.path.getsize(path)
    test_time = get_test_time(t0,tf)
    add_test_data('write','hdf5_tb',size_read,test_time,**kwargs)

def write_hdf5_pd(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_pd.hdf5"
    print(f"\t\t\t[write_hdf5_pd] {path}")

    if os.path.isfile(path): os.remove(path)

    X_ = pd.DataFrame(X)

    t0 = datetime.now()
    X_.to_hdf(path, 'X', index=False)
    tf = datetime.now()

    size_read = os.path.getsize(path)
    test_time = get_test_time(t0,tf)
    add_test_data('write','hdf5_pd',size_read,test_time,**kwargs)

# =============================================================================
# test functions
# =============================================================================

def test_read(*args,**kwargs):
    basename = args[0]
    print(f"\t\t[test_read] {basename}")
    read_hdf5_h5(*args,**kwargs)
    read_hdf5_tb(*args,**kwargs)
    read_hdf5_pd(*args,**kwargs)

def test_write(X,*args,**kwargs):
    shape = args[1]
    size_write_MB = args[-1]
    print(f"\t\t[test_write] shape: {X.shape},size (MB): {size_write_MB}")
    write_hdf5_h5(X,*args,**kwargs)
    write_hdf5_tb(X,*args,**kwargs)
    write_hdf5_pd(X,*args,**kwargs)

def get_test_time(t0,tf):
    dt = (tf - t0)
    return dt.total_seconds()

def add_test_data(operation,method,size_read,test_time,shape=None,replicant=None,size_write_MB=None):
    loc = (operation,method,shape,replicant)
    data = [size_read,size_write_MB,test_time]
    test.data.loc[loc] = data

# =============================================================================
# plot
# =============================================================================

def generate_plots():
    print("[generate_plots]")

    fig, ax = plt.subplots(figsize=(15,8), nrows=2, ncols=1, sharex=False, sharey=False)

    for i,operation in enumerate(test.operations):

        for method, color in test.colors.items():

            data = test.data.loc[(operation,method)].copy()
            data.sort_values('size_read_MB', inplace=True)
            ax[i].scatter(data.size_read_MB, data.rate, alpha=0.5, color=color)

            data_ = data.groupby(level='shape').mean()
            data_.sort_values('size_read_MB', inplace=True)
            ax[i].plot(data_.size_read_MB, data_.rate, color=color, label=method)

        ax[i].set_title(operation.title())
        ax[i].set_xlabel('File Size [MB]')
        ax[i].set_ylabel('Rate [MB/s]')

        _,xmax = ax[i].get_xlim()
        _,ymax = ax[i].get_ylim()
        ax[i].set_xlim([0,xmax])
        ax[i].set_ylim([0,ymax])

        ax[i].legend()

    plt.tight_layout()

    fig.savefig('test_results.pdf', dpi=300)

# =============================================================================
# main
# =============================================================================

def main():
    print("[main]")

    os.chdir('data_test')

    shutil.rmtree('data', ignore_errors=True)
    os.mkdir('data')

    for shape in test.shapes:
        print(f"TEST: Shape {shape}")
        for n in test.replicants:
            print(f"\tTEST: Replicant {n}")

            Nrows, Ncols = shape
            Ncells = Nrows * Ncols

            X = np.random.randn(*shape)
            size_write_MB = X.nbytes * BYTE_TO_MB
            basename = os.path.join('data', f"{X.dtype.name}_{Nrows}_{Ncols}")

            args = (basename,Nrows,Ncols)
            kwargs = dict(shape = shape,
                          replicant = n,
                          size_write_MB = size_write_MB,
                          )

            test_write(X,*args,**kwargs)
            test_read(*args,**kwargs)

    test.data['size_read_MB'] = test.data['size_read'] * BYTE_TO_MB
    test.data['rate'] = test.data.size_write_MB / test.data.test_time
    generate_plots()

if "__main__" == __name__:
    main()
