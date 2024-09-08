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
    shapes = list(itertools.product(opts.rows,opts.columns))
    columns = ['size_MB','size','test_time']
    read_times = {'npy_np' : [],
                  'csv_pd' : [],
                  'xlsx_pd': [],
                  'h5df_h5': [],
                  'h5df_tb': [],
                  'h5df_pd': [],
                  'pkl_pkl': [],
                  'ds_sql' : [],
                  }
    write_times = {'npy_np': [],
                  'csv_pd' : [],
                  'xlsx_pd': [],
                  'h5df_h5': [],
                  'h5df_tb': [],
                  'h5df_pd': [],
                  'pkl_pkl': [],
                  'ds_sql' : [],
                  }
    colors = {'npy_np' : 'r',
              'csv_pd' : 'orange',
              'xlsx_pd': 'y',
              'h5df_h5': '#00ff00',
              'h5df_tb': '#00cc00',
              'h5df_pd': '#009900',
              'pkl_pkl': 'b',
              'ds_sql' : 'c',
              }

# =============================================================================
# read functions
# =============================================================================

def read_npy(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.npy"
    print(f"\t\t[read_npy] {path}")

    t0 = datetime.now()
    X_ = np.load(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.read_times['npy_np'].append([size_MB,size,test_time])

def read_csv_pd(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.csv"
    print(f"\t\t[read_csv_pd] {path}")

    t0 = datetime.now()
    X_ = pd.read_csv(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.read_times['csv_pd'].append([size_MB,size,test_time])

def read_xlsx_pd(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.xlsx"
    print(f"\t\t[read_xlsx_pd] {path}")

    t0 = datetime.now()
    X_ = pd.read_excel(path)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.read_times['xlsx_pd'].append([size_MB,size,test_time])

def read_hdf5_h5(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}_h5.h5df"
    print(f"\t\t[read_hdf5_h5] {path}")

    t0 = datetime.now()
    with h5.File(path,'r') as f:
        X_ = f['X']
        size = X_.size
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.read_times['h5df_h5'].append([size_MB,size,test_time])

def read_h5df_tb(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}_tb.h5df"
    print(f"\t\t[read_h5df_tb] {path}")

    t0 = datetime.now()
    with tb.open_file(path, mode='r') as f:
        X_ = f.root.X.read()
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.read_times['h5df_tb'].append([size_MB,size,test_time])

def read_h5df_pd(basename,shape,Nrows,Ncols,size,size_MB):
    pass

def read_pkl_pkl(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.pkl"
    print(f"\t\t[read_pkl_pkl] {path}")

    t0 = datetime.now()
    with open(path,'rb') as f:
        X_ = pkl.load(f)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.read_times['pkl_pkl'].append([size_MB,size,test_time])

def read_ds_sql(basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.db"
    print(f"\t\t[read_ds_sql] {path}")

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
    test.read_times['ds_sql'].append([size_MB,size,test_time])

# =============================================================================
# write functions
# =============================================================================

def write_npy(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.npy"
    print(f"\t\t[write_npy] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    np.save(path,X)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.write_times['npy_np'].append([size_MB,size,test_time])

def write_csv_pd(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.csv"
    print(f"\t\t[write_csv_pd] {path}")

    if os.path.isfile(path): os.remove(path)

    X_ = pd.DataFrame(X)

    t0 = datetime.now()
    X_.to_csv(path, index=False)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.write_times['csv_pd'].append([size_MB,size,test_time])

def write_xlsx_pd(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.xlsx"
    print(f"\t\t[write_xlsx_pd] {path}")

    if os.path.isfile(path): os.remove(path)

    X_ = pd.DataFrame(X)

    t0 = datetime.now()
    X_.to_excel(path, index=False)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.write_times['xlsx_pd'].append([size_MB,size,test_time])

def write_h5df_h5(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}_h5.h5df"
    print(f"\t\t[write_h5df_h5] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    # TODO: support parallel HDF5?
    # TODO: support virtual datasets?
    with h5.File(path,'w') as f:
        X_ = f.create_dataset('X', data=X)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.write_times['h5df_h5'].append([size_MB,size,test_time])

def write_h5df_tb(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}_tb.h5df"
    print(f"\t\t[write_h5df_tb] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    with tb.open_file(path,'w') as f:
        ds = f.create_array(f.root, 'X', X)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.write_times['h5df_tb'].append([size_MB,size,test_time])

def write_h5df_pd(X,basename,shape,Nrows,Ncols,size,size_MB):
    pass

def write_pkl_pkl(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.pkl"
    print(f"\t\t[write_pkl_pkl] {path}")

    if os.path.isfile(path): os.remove(path)

    t0 = datetime.now()
    with open(path,'wb') as f:
        pkl.dump(X,f)
    tf = datetime.now()

    test_time = get_test_time(t0,tf)
    test.write_times['pkl_pkl'].append([size_MB,size,test_time])

def write_ds_sql(X,basename,shape,Nrows,Ncols,size,size_MB):
    path = f"{basename}.db"
    print(f"\t\t[write_ds_sql] {path}")

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
    test.write_times['ds_sql'].append([size_MB,size,test_time])

# =============================================================================
# test functions
# =============================================================================

def test_read(*args):
    basename = args[0]
    print(f"\t[test_read] {basename}")
    read_npy(*args)
    read_csv_pd(*args)
    read_xlsx_pd(*args)
    read_hdf5_h5(*args)
    read_h5df_tb(*args)
    read_h5df_pd(*args)
    read_pkl_pkl(*args)
    read_ds_sql(*args)

def test_write(X,*args):
    shape = args[1]
    size_MB = args[-1]
    print(f"\t[test_write] shape: {X.shape},size (MB): {size_MB}")
    write_npy(X,*args)
    write_csv_pd(X,*args)
    write_xlsx_pd(X,*args)
    write_h5df_h5(X,*args)
    write_h5df_tb(X,*args)
    write_h5df_pd(X,*args)
    write_pkl_pkl(X,*args)
    write_ds_sql(X,*args)

def get_test_time(t0,tf):
    dt = (tf - t0)
    return dt.total_seconds()

def convert_test_data_to_dataframes(key):
    print(f"[convert_test_data_to_dataframes] {key}")

    test_ = getattr(test, key)

    for key,val in test_.items():
        df = pd.DataFrame(val,columns=test.columns)
        df['rate'] = df.size_MB / df.test_time
        df.sort_values('size', inplace=True)
        test_[key] = df

# =============================================================================
# plot
# =============================================================================

def generate_plot():
    print("[generate_plot]")

    fig, ax = plt.subplots(figsize=(15,9), nrows=2, ncols=1, sharex=True, sharey=False)

    ax[0].set_title('Read')

    for key, color in test.colors.items():

        kwargs = dict(color=color,
                      label=key,
                      alpha=0.5,
                      )

        read_df = test.read_times[key]
        ax[0].plot(read_df['size'], read_df.rate, **kwargs)

        write_df = test.write_times[key]
        ax[1].plot(write_df['size'], write_df.rate, **kwargs)

    for i in range(2):

        ax[i].set_xlabel('Array Size')
        ax[i].set_ylabel('Rate [MB/s]')

        _,xmax = ax[i].get_xlim()
        _,ymax = ax[i].get_ylim()
        ax[i].set_xlim([0,xmax])
        ax[i].set_ylim([0,ymax])

    plt.tight_layout()
    ax[0].legend()

    fig.savefig('test_results.pdf', dpi=300)

# =============================================================================
# main
# =============================================================================

def main():
    print("[main]")

    os.chdir('data_test')

    for shape in test.shapes:
        print(f"TEST: Nrows,Ncols: {shape}")

        Nrows, Ncols = shape
        size = Nrows * Ncols

        X = np.random.randn(*shape)
        size_MB = X.nbytes * BYTE_TO_MB
        basename = os.path.join('data', f"{X.dtype.name}_{Nrows}_{Ncols}")

        args = (basename, shape, Nrows, Ncols, size, size_MB)

        test_write(X,*args)
        test_read(*args)

    convert_test_data_to_dataframes('read_times')
    convert_test_data_to_dataframes('write_times')
    generate_plot()

if "__main__" == __name__:
    main()
