import os

def read_data_from_csv(read_file, write_file):
    assert os.path.exists(read_file + "/data.txt"), "Please follow the 'PyKT' guidelines from README.md before launching PyKT"
    print("MIAAM's dataset was already processed using `notebooks/pykt_processing`, nothing else to do")
    return
