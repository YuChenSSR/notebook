import pandas as pd
import fire

def read_csi(csi_path):
    return pd.read_csv(csi_path, sep='\t', header=None)
    
    


def main(
    csi_path = "/home/idc2/notebook/qlib_bin/cn_data_train/instruments/csi800b.txt"
):
    
    csi_file = read_csi(csi_path)
    csi_file.columns = ['instrument', 'start_date', 'end_date']

    csi_file['start_date'] = pd.to_datetime(csi_file['start_date'])
    csi_file = csi_file[(csi_file['start_date'] <= '2020-01-01') & (csi_file['end_date'] == '2025-12-08')]


    csi_filename = f"./csi800c.txt"
    csi_file.to_csv(csi_filename, index=False, header=False, sep='\t')
    
    print(csi_file)

if __name__ == "__main__":
    fire.Fire(main)