import zipfile
import pathlib


if __name__ == "__main__":
    zips_folder_path = pathlib.Path('C:\\Users\\User\\Documents\\UFMG\\POC1\\world_graphs\\zipped\\')
    unziped_folder_target_path = pathlib.Path('C:\\Users\\User\\Documents\\UFMG\\POC1\\world_graphs\\')
    zip_files = list(zips_folder_path.glob('*.zip'))

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unziped_folder_target_path)