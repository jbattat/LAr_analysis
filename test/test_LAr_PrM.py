import sys
sys.path.append("../")
import LAr_PrM as pm

dataDir = pm.data_dir()
print(f"dataDir = {dataDir}")

meta = pm.get_run_metadata()

fname = '20250522T130458.csv'
hvc = pm.get_hv_of_fname(fname, meta, source='c')
hvag = pm.get_hv_of_fname(fname, meta, source='ag')
hva = pm.get_hv_of_fname(fname, meta, source='a')
hvall = pm.get_hv_of_fname(fname, meta, source='all')
print(hvc)
print(hvag)
print(hva)
print(hvall)
