import numpy as np
import pandas as pd
import h5py as h5
import tables as tb
import pickle as pkl
import sqlite3 as sql

import argparse
import itertools
import matplotlib.pyplot as plt
import os

from datetime import datetime

# =============================================================================
# run options
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('-col','--columns',
                    nargs = "*",
                    default = (10 ** np.arange(4)).tolist(),
                    type = int,
                    help = "the number of column(s) to use")

parser.add_argument('-row', '--rows',
                    nargs = "*",
                    default = (10 ** np.arange(6)).tolist(),
                    type = int,
                    help = "the number of row(s) to use")

parser.add_argument('-rep', '--replicants',
                    default = 10,
                    type = int,
                    help = "the number of times to repeat each experiment")

opts = parser.parse_args()

# =============================================================================
# global
# =============================================================================

BYTE_TO_MB = 1 / np.square(1024).item()

class test:
    operations = ['read','write']
    colors = {'npy_np' : 'r',
              'csv_pd' : 'orange',
              'xlsx_pd': 'y',
              'h5df_h5': '#00ff00',
              'h5df_tb': '#00cc00',
              'h5df_pd': '#009900',
              'pkl_pkl': 'b',
              'ds_sql' : 'c',
              'ds_sql' : 'm', # TODO
              }
    methods = list(colors.keys())
    shapes = list(itertools.product(opts.rows,opts.columns))
    replicants = range(opts.replicants)
    columns = ['Ncells',
               'size_MB',
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

def read_npy(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.npy"
    print(f"\t\t\t[read_npy] {path}")

    t0 = datetime.now()
    X_ = np.load(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','npy_np',test_time,**kwargs)

def read_csv_pd(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.csv"
    print(f"\t\t\t[read_csv_pd] {path}")

    t0 = datetime.now()
    X_ = pd.read_csv(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','csv_pd',test_time,**kwargs)

def read_xlsx_pd(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.xlsx"
    print(f"\t\t\t[read_xlsx_pd] {path}")

    t0 = datetime.now()
    X_ = pd.read_excel(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','xlsx_pd',test_time,**kwargs)

def read_h5df_h5(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_h5.h5df"
    print(f"\t\t\t[read_h5df_h5] {path}")

    t0 = datetime.now()
    with h5.File(path,'r') as f:
        X_ = f['X']
        size = X_.size
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','h5df_h5',test_time,**kwargs)

def read_h5df_tb(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_tb.h5df"
    print(f"\t\t\t[read_h5df_tb] {path}")

    t0 = datetime.now()
    with tb.open_file(path, mode='r') as f:
        X_ = f.root.X.read()
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','h5df_tb',test_time,**kwargs)

def read_h5df_pd(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_pd.h5df"
    print(f"\t\t\t[read_h5df_pd] {path}")

    t0 = datetime.now()
    X_ = pd.read_hdf(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','h5df_pd',test_time,**kwargs)

def read_pkl_pkl(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.pkl"
    print(f"\t\t\t[read_pkl_pkl] {path}")

    t0 = datetime.now()
    with open(path,'rb') as f:
        X_ = pkl.load(f)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','pkl_pkl',test_time,**kwargs)

def read_ds_sql(basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.db"
    print(f"\t\t\t[read_ds_sql] {path}")

    t0 = datetime.now()
    cx = sql.connect(path)
    cursor = cx.cursor()
    cmd = 'SELECT ' + ', '.join([f"col{x}" for x in range(Ncols)]) + ' FROM X'
    cursor.execute(cmd)
    rows = cursor.fetchall()
    X_ = np.array(rows)
    cx.close()
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('read','ds_sql',test_time,**kwargs)

# =============================================================================
# write functions
# =============================================================================

def write_npy(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.npy"
    print(f"\t\t\t[write_npy] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    np.save(path,X)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','npy_np',test_time,**kwargs)

def write_csv_pd(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.csv"
    print(f"\t\t\t[write_csv_pd] {path}")

    if os.path.isfile(path): os.remove(path)

    X_ = pd.DataFrame(X)

    t0 = datetime.now()
    X_.to_csv(path, index=False)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','csv_pd',test_time,**kwargs)

def write_xlsx_pd(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.xlsx"
    print(f"\t\t\t[write_xlsx_pd] {path}")

    if os.path.isfile(path): os.remove(path)

    X_ = pd.DataFrame(X)

    t0 = datetime.now()
    X_.to_excel(path, index=False)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','xlsx_pd',test_time,**kwargs)

def write_h5df_h5(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_h5.h5df"
    print(f"\t\t\t[write_h5df_h5] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    # TODO: support parallel h5df?
    # TODO: support virtual datasets?
    with h5.File(path,'w') as f:
        X_ = f.create_dataset('X', data=X)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','h5df_h5',test_time,**kwargs)

def write_h5df_tb(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_tb.h5df"
    print(f"\t\t\t[write_h5df_tb] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    with tb.open_file(path,'w') as f:
        ds = f.create_array(f.root, 'X', X)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','h5df_tb',test_time,**kwargs)

def write_h5df_pd(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}_pd.h5df"
    print(f"\t\t\t[write_h5df_pd] {path}")

    if os.path.isfile(path): os.remove(path)

    X_ = pd.DataFrame(X)

    t0 = datetime.now()
    X_.to_hdf(path, 'X', index=False)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','h5df_pd',test_time,**kwargs)

def write_pkl_pkl(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.pkl"
    print(f"\t\t\t[write_pkl_pkl] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    with open(path,'wb') as f:
        pkl.dump(X,f)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','pkl_pkl',test_time,**kwargs)

def write_ds_sql(X,basename,Nrows,Ncols,**kwargs):
    path = f"{basename}.db"
    print(f"\t\t\t[write_ds_sql] {path}")

    if os.path.isfile(path): os.remove(path)

    Nrows,Ncols = X.shape

    t0 = datetime.now()
    cx = sql.connect(path)
    cursor = cx.cursor()
    cmd = 'CREATE TABLE IF NOT EXISTS X (' + ', '.join([f'col{x} FLOAT' for x in range(Ncols)]) + ')'
    cursor.execute(cmd)
    cmd_ = 'INSERT INTO X (' + ', '.join([f'col{x}' for x in range(Ncols)]) + ') VALUES (' + ', '.join(['?' for _ in range(Ncols)]) + ')'
    for x in X:
        cursor.execute(cmd_, tuple(x))
    cx.commit()
    cx.close()
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    add_test_data('write','ds_sql',test_time,**kwargs)

# =============================================================================
# test functions
# =============================================================================

def test_read(*args,**kwargs):
    basename = args[0]
    print(f"\t\t[test_read] {basename}")
    read_npy(*args,**kwargs)
    read_csv_pd(*args,**kwargs)
    read_xlsx_pd(*args,**kwargs)
    read_h5df_h5(*args,**kwargs)
    read_h5df_tb(*args,**kwargs)
    read_h5df_pd(*args,**kwargs)
    read_pkl_pkl(*args,**kwargs)
    read_ds_sql(*args,**kwargs)

def test_write(X,*args,**kwargs):
    shape = args[1]
    size_MB = args[-1]
    print(f"\t\t[test_write] shape: {X.shape},size (MB): {size_MB}")
    write_npy(X,*args,**kwargs)
    write_csv_pd(X,*args,**kwargs)
    write_xlsx_pd(X,*args,**kwargs)
    write_h5df_h5(X,*args,**kwargs)
    write_h5df_tb(X,*args,**kwargs)
    write_h5df_pd(X,*args,**kwargs)
    write_pkl_pkl(X,*args,**kwargs)
    write_ds_sql(X,*args,**kwargs)

def get_test_time(t0,tf):
    dt = (tf - t0)
    return dt.total_seconds()

def add_test_data(operation,method,test_time,shape=None,replicant=None,Ncells=None,size_MB=None):
    loc = (operation,method,shape,replicant)
    data = [Ncells,size_MB,test_time]
    test.data.loc[loc] = data

# =============================================================================
# plot
# =============================================================================

def generate_plots():
    print("[generate_plots]")

    fig, ax = plt.subplots(figsize=(15,9), nrows=2, ncols=2, sharex=False, sharey=False)

    for i,operation in enumerate(test.operations):

        for method, color in test.colors.items():

            data = test.data.loc[(operation,method)]
            ax[i,0].scatter(data.size_MB, data.test_time, alpha=0.5, color=color)
            ax[i,1].scatter(data.size_MB, data.Ncells, alpha = 0.5, color=color)

            data_ = data.groupby(level='shape').mean()
            ax[i,0].plot(data_.size_MB, data_.test_time, color=color, label=method)
            ax[i,1].plot(data_.size_MB, data_.Ncells, color=color, label=method)

        ax[i,0].set_ylabel('Time [s]')
        ax[i,1].set_xlabel('Cell Size')

        for j in range(2):

            ax[i,j].set_title(operation)
            ax[i,j].set_xlabel('Size [MB]')

            _,xmax = ax[i,j].get_xlim()
            _,ymax = ax[i,j].get_ylim()
            ax[i,j].set_xlim([0,xmax])
            ax[i,j].set_ylim([0,ymax])

            ax[i,j].legend()

    plt.tight_layout()

    fig.savefig('test_results.pdf', dpi=300)

# =============================================================================
# main
# =============================================================================

def main():
    print("[main]")

    os.chdir('data_test')

    for shape in test.shapes:
        print(f"TEST: Shape {shape}")
        for n in test.replicants:
            print(f"\tTEST: Replicant {n}")

            Nrows, Ncols = shape
            Ncells = Nrows * Ncols

            X = np.random.randn(*shape)
            size_MB = X.nbytes * BYTE_TO_MB
            basename = os.path.join('data', f"{X.dtype.name}_{Nrows}_{Ncols}")

            args = (basename,Nrows,Ncols)
            kwargs = dict(shape = shape,
                          replicant = n,
                          Ncells = Ncells,
                          size_MB = size_MB,
                          )

            test_write(X,*args,**kwargs)
            test_read(*args,**kwargs)

    test.data['rate'] = test.data.size_MB / test.data.test_time
    generate_plots()

if "__main__" == __name__:
    main()
