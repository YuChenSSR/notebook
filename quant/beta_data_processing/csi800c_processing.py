import pandas as pd
import fire

def read_csi(csi_path):
    return pd.read_csv(csi_path, sep='\t', header=None)
    

def main(
    csi_path = "/home/a/notebook/cn_data_train/instruments",
):
    
    
    csi_file = read_csi(f"{csi_path}/csi800b.txt")
    csi_file.columns = ['instrument', 'start_date', 'end_date']

    csi_file['start_date'] = pd.to_datetime(csi_file['start_date'])
    csi_file = csi_file[(csi_file['start_date'] <= '2020-01-01') & (csi_file['end_date'] == csi_file['end_date'].max())]



    csi_c_filename = f"{csi_path}/csi800c.txt"
    csi_file.to_csv(csi_c_filename, index=False, header=False, sep='\t')
    
if __name__ == "__main__":
    fire.Fire(main)