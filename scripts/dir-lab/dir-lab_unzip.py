import os
from tqdm import tqdm
import zipfile

def main():
    # dir-lab data directory
    DATA_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/DIR_LAB_4DCT/zip"
    OUTPUT_DIR = "/home/jacky/DIIR-JK-NAS/data/lung_data/DIR_LAB_4DCT/unzip"

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    pbar = tqdm(os.listdir(DATA_DIR))
    for file in pbar:
        pbar.set_description(file)
        try:
            with zipfile.ZipFile(os.path.join(DATA_DIR,file),"r") as zip_ref:
                zip_ref.extractall(os.path.join(OUTPUT_DIR))

            tqdm.write("unzip {} success".format(os.path.join(DATA_DIR,file)))
        except:
            tqdm("unzip {} fail".format(os.path.join(DATA_DIR,file)))
            continue

if __name__ == "__main__":
    main()