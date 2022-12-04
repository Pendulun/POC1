import pandas as pd
import osmnx
import networkx
import pathlib
import multiprocessing

def complete_with_basic_info(graph_path:pathlib.Path, features:dict) -> dict:
    city_name, city_id = get_name_and_id(graph_path)
    features['city_name'] = city_name
    features['city_id'] = city_id
    return features

def get_name_and_id(graph_path:pathlib.Path):
    file_name = graph_path.stem
    file_name_parts = file_name.split('-')
    city_name = " ".join(file_name_parts[0].split('_'))
    city_id = file_name_parts[1]
    return city_name, city_id

def compute_osmnx_features(graph_path:pathlib.Path) -> dict:
    return dict()

def compute_features(graph_path:pathlib.Path) -> dict:
    full_features = dict()

    full_features = complete_with_basic_info(graph_path, full_features)

    osmnx_features = compute_osmnx_features(graph_path)

    full_features = {**full_features, **osmnx_features}
    return full_features

def main():
    graph_folder_path = pathlib.Path('C:\\Users\\User\\Documents\\UFMG\\POC1\\world_graphs\\')

    if not graph_folder_path.exists():
        print(f'{graph_folder_path} does not exists!')
        exit(-1)
    
    if not graph_folder_path.is_dir():
        print(f"{graph_folder_path} is not a folder!")
        exit(-1)
    
    graph_files = list(graph_folder_path.glob('*.graphml'))[:3]

    features = [compute_features(graph_path) for graph_path in graph_files]

    features_df = pd.DataFrame.from_records(features)

    new_features_folder_path = pathlib.Path('./new_features')

    new_features_folder_path.mkdir(exist_ok=True)

    features_df.to_csv(new_features_folder_path / 'new_features.csv', index=False)

if __name__ == "__main__":
    main()