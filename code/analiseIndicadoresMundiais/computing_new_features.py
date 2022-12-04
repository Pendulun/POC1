import pandas as pd
import osmnx as ox
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

def compute_osmnx_features(G) -> dict:
    #https://github.com/gboeing/osmnx-examples/blob/main/notebooks/06-stats-indicators-centrality.ipynb
    stats = ox.basic_stats(G)

    for k, count in stats["streets_per_node_counts"].items():
        stats["{}way_int_count".format(k)] = count

    for k, proportion in stats["streets_per_node_proportions"].items():
        stats["{}way_int_prop".format(k)] = proportion

    # delete the no longer needed dict elements
    del stats["streets_per_node_counts"]
    del stats["streets_per_node_proportions"]
    return stats

def remove_features_already_computed(full_features:dict):
    unwanted_features = [
        'circuity_avg', 'intersection_count', 'k_avg', 'edge_length_total',
        'edge_length_avg', 'street_length_total', 'street_length_avg', 
        '0way_int_prop', '1way_int_prop', '4way_int_prop', '3way_int_prop',
        'clean_intersection_count', 'n', 'm', 'self_loop_proportion',
        'street_segment_count', '0way_int_count'

    ]
    features_list = full_features.keys()
    for feature in unwanted_features:
        if feature in features_list:
            del full_features[feature]
    
    return full_features

def compute_features(graph_path:pathlib.Path) -> dict:
    full_features = dict()
    full_features = complete_with_basic_info(graph_path, full_features)
    try:
        G = ox.load_graphml(graph_path)

        #Falta conseguir a área em metros quadrados para computar algumas outras features
        #Para isso, devo ler também o arquivo de features já existentes.
        osmnx_features = compute_osmnx_features(G)

        full_features = {**full_features, **osmnx_features}
        full_features = remove_features_already_computed(full_features)
    except Exception as e:
        print(f"ERRO: {full_features['city_name']}.\n{e}")
    else:
        print(f"Completou {full_features['city_name']}")
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
    features_df.fillna(0, inplace=True)

    new_features_folder_path = pathlib.Path('./new_features')

    new_features_folder_path.mkdir(exist_ok=True)

    features_df.to_csv(new_features_folder_path / 'new_features.csv', index=False)

if __name__ == "__main__":
    main()