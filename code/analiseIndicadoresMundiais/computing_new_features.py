import pandas as pd
import osmnx as ox
import networkx as nx
import pathlib
import multiprocessing
import warnings
from timeit import default_timer as timer

GRAPH_TYPES = {'small':1, 'big':2, 'target':3}

def get_num_processes_used(TARGET_GRAPH_TYPE):
    if TARGET_GRAPH_TYPE == GRAPH_TYPES['small']:
        processes = (multiprocessing.cpu_count() // 2) + 1
    elif TARGET_GRAPH_TYPE == GRAPH_TYPES['big']:
        processes = 1
    elif TARGET_GRAPH_TYPE == GRAPH_TYPES['target']:
        processes = 1
    return processes

def get_graphml_file_paths(graph_type:int) -> list:
    graph_folder_path = pathlib.Path('C:\\Users\\User\\Documents\\UFMG\\POC1\\world_graphs\\')
    check_valid_folder(graph_folder_path)

    all_graph_files = graph_folder_path.glob('*.graphml')
    big_graphs_cities_ids = get_big_graphs_cities_ids()
    target_graph_files = list()

    if graph_type == GRAPH_TYPES['small']:
        target_graph_files = [
            file for file in all_graph_files if get_city_id_from_filename(file) not in big_graphs_cities_ids
            ]
    
    elif graph_type == GRAPH_TYPES['big']:
        #Because of memory restrictions, cities with 300_000 plus nodes cant be processed
        #These are: tokyo, jakarta, osaka_kyoto, nagoya, mexico_city
        unprocessable_cities_path = pathlib.Path('./data/unprocessable_cities.csv')
        unprocessable_cities_df = pd.read_csv(unprocessable_cities_path)
        unprocessable_cities_ids = unprocessable_cities_df['uc_id'].values
        for file in all_graph_files:
            city_id = get_city_id_from_filename(file)
            if city_id not in unprocessable_cities_ids and city_id in big_graphs_cities_ids:
                target_graph_files.append(file)

    elif graph_type == GRAPH_TYPES['target']:
        #This is for when I want to process specific cities
        target_cities_ids_as_ints = []
        target_graph_files = [
            file for file in all_graph_files if get_city_id_from_filename(file) in target_cities_ids_as_ints
            ]
    
    return target_graph_files

def get_city_id_from_filename(filename:str) -> int:
    return int(filename.stem.split('-')[1])

def check_valid_folder(graph_folder_path):
    if not graph_folder_path.exists():
        print(f'{graph_folder_path} does not exists!')
        exit(-1)
    
    if not graph_folder_path.is_dir():
        print(f"{graph_folder_path} is not a folder!")
        exit(-1)

def get_big_graphs_cities_ids() -> set:
    big_graphs_indicators_path = pathlib.Path("./data/grafos_grandes.csv")
    big_graphs_df = pd.read_csv(big_graphs_indicators_path)
    big_graphs_cities_ids = set(big_graphs_df['uc_id'].values)
    return big_graphs_cities_ids

def get_indicators_df():
    indicators_from_paper_path = pathlib.Path('./data/indicators.csv')
    if not indicators_from_paper_path.exists():
        print(f"{indicators_from_paper_path} does not exists!")
        exit(-1)
    
    indicators_df = pd.read_csv(indicators_from_paper_path)
    return indicators_df

def get_save_step(graph_type:int):
    if graph_type == GRAPH_TYPES['small']:
        return 20
    elif graph_type == GRAPH_TYPES['big']:
        return 20
    elif graph_type == GRAPH_TYPES['target']:
        return 100

def compute_features(graph_path:pathlib.Path, indicators_df:pd.DataFrame) -> dict:
    #So to ignore networkx warnings
    warnings.filterwarnings("ignore")
    full_features = dict()
    full_features = complete_with_basic_info(graph_path, full_features)
    try:
        G = ox.load_graphml(graph_path)

        osmnx_features = compute_osmnx_features(G, full_features['city_id'], indicators_df)
        other_calculated_features = compute_other_features(osmnx_features)
        networkx_features = compute_networkx_features(G)

        all_features_list = [full_features, osmnx_features, other_calculated_features, networkx_features]
        full_features = merge_all_features(all_features_list)
    except Exception as e:
        print(f"ERRO: {full_features['city_name']}.\n{e}")
    else:
        print(f"Completou {full_features['city_name']}")
        return full_features

def complete_with_basic_info(graph_path:pathlib.Path, features:dict) -> dict:
    city_name, city_id = get_name_and_id(graph_path)
    features['city_name'] = city_name
    features['city_id'] = int(city_id)
    return features

def get_name_and_id(graph_path:pathlib.Path):
    file_name = graph_path.stem
    file_name_parts = file_name.split('-')
    city_name = " ".join(file_name_parts[0].split('_'))
    city_id = file_name_parts[1]
    return city_name, city_id

def compute_osmnx_features(G, city_id:str, indicators_df:pd.DataFrame) -> dict:
    #https://github.com/gboeing/osmnx-examples/blob/main/notebooks/06-stats-indicators-centrality.ipynb
    city_build_square_km_area = float(indicators_df[indicators_df['uc_id'] == city_id]['built_up_area'].values[0])
    city_build_square_m_area = city_build_square_km_area * 1_000_000
    stats = ox.basic_stats(G, area=city_build_square_m_area)

    for k, count in stats["streets_per_node_counts"].items():
        stats["{}way_int_count".format(k)] = count

    for k, proportion in stats["streets_per_node_proportions"].items():
        stats["{}way_int_prop".format(k)] = proportion

    # delete the no longer needed dict elements
    del stats["streets_per_node_counts"]
    del stats["streets_per_node_proportions"]
    return stats

def compute_other_features(features:dict) -> dict:
    other_features = dict()
    other_features['nodes_per_km_street'] = features['n'] / features['street_length_total']
    other_features['organic_prop'] = (features['1way_int_count']+features['3way_int_count'])/(features['n']-features['2way_int_count'])
    other_features['meshedness_coefficient'] = (features['m']-features['n']+1)/((2*features['n']*(1-features['2way_int_prop']))-5)
    return other_features

def compute_networkx_features(G) -> dict:
    return dict()
    features = dict()
    undirected_graph = nx.Graph(G)
    greatest_undirected_component_nodes = max(nx.connected_components(undirected_graph), key=len)
    # greatest_directed_component_graph = G.subgraph(greatest_undirected_component_nodes)
    greatest_undirected_component_graph = undirected_graph.subgraph(greatest_undirected_component_nodes)
    #So to free memory
    greatest_undirected_component_nodes = None

    information_cent = nx.information_centrality(greatest_undirected_component_graph)
    features['max_info_centrality'] = max(list(information_cent.values()))
    features['avg_info_centrality'] = features['max_info_centrality']/len(information_cent)

    #So to free memory
    greatest_undirected_component_graph = None

    #Too expensive to compute
    # features['node_conectivity'] = len(nx.minimum_node_cut(greatest_component))
    # features['edge_conectivity'] = len(nx.minimum_edge_cut(greatest_component))
    # features['avg_conectivity'] = nx.average_node_connectivity(undirected_graph)
    
    betweenness_cent:dict = nx.betweenness_centrality(G)
    features['max_betweenness_centrality'] = max(list(betweenness_cent.values()))
    features['avg_betweenness_centrality'] = features['max_betweenness_centrality']/len(betweenness_cent)
    
    features['global_efficiency'] = nx.global_efficiency(undirected_graph)

    #This method already treats for multiple components.
    #See it's documentation for the Wasserman and Faust version
    closeness_cent = nx.closeness_centrality(G)
    features['avg_closeness_centrality'] = max(list(closeness_cent.values()))/len(closeness_cent)
    return features

def merge_all_features(features_dict_list:list) -> dict:
    full_features = features_dict_list[0]
    for features in features_dict_list[1:]:
        full_features = {**full_features, **features}
    
    return full_features

def clean_features(features):
    cleaned_features = list()
    for feature in features:
        cleaned_features.append(remove_features_already_computed(feature))
    
    return cleaned_features

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

def save_features(features:list, graph_type:int, batch_count:int):

    features_df = pd.DataFrame.from_records(features)
    features_df.fillna(0, inplace=True)
    new_features_folder_path = pathlib.Path('./new_features')
    new_features_folder_path.mkdir(exist_ok=True)

    if graph_type == GRAPH_TYPES['small']:
        file_name = f'small_graphs_{batch_count}.csv'
    elif graph_type == GRAPH_TYPES['big']:
        file_name = f"big_graphs_{batch_count}.csv"
    elif graph_type == GRAPH_TYPES['target']:
        file_name = f"target_graphs_{batch_count}.csv"

    features_df.to_csv(new_features_folder_path / file_name, index=False)

def main():
    TARGET_GRAPH_TYPE = GRAPH_TYPES['big']
    print(f"GRAPH TARGET TYPE: {TARGET_GRAPH_TYPE}")
    
    PROCESSES = get_num_processes_used(TARGET_GRAPH_TYPE)
    print(f'PROCESSES USED: {PROCESSES}')

    CHUNKSIZE = 1
    print(f'CHUNKSIZE USED: {CHUNKSIZE}')

    graph_files = get_graphml_file_paths(TARGET_GRAPH_TYPE)
    indicators_df = get_indicators_df()

    min_batch_size = get_save_step(TARGET_GRAPH_TYPE)
    print(f"BATCH MIN SIZE: {min_batch_size}")
    STARTING_BATCH_ID = 6
    print(f"STARTING BATCH: {STARTING_BATCH_ID}")
    starting_batch_idx = STARTING_BATCH_ID*min_batch_size
    NUM_BATCHES_TO_RUN = 20
    print(f"NUM BATCHES TO RUN: {NUM_BATCHES_TO_RUN}")

    curr_batch_id = STARTING_BATCH_ID
    num_batch = 0
    elapsed_time_per_batch = list()
    while starting_batch_idx < len(graph_files) and num_batch < NUM_BATCHES_TO_RUN:
        graph_path_batch = graph_files[starting_batch_idx: starting_batch_idx+min_batch_size]
        try:
            print(f"Começando batch {curr_batch_id}")
            time_start = timer()
            with multiprocessing.Pool(PROCESSES) as pool:
                params = [(graph_path, indicators_df) for graph_path in graph_path_batch]
                features = pool.starmap(compute_features, params, chunksize=CHUNKSIZE)
            time_end = timer()
            elapsed_time_per_batch.append(time_end - time_start)
        except Exception as e:
            print(f"Error on batch {curr_batch_id}!\n {e}")
        else:
            features = clean_features(features)
            save_features(features, TARGET_GRAPH_TYPE, curr_batch_id)
        finally:
            starting_batch_idx += min_batch_size
            curr_batch_id += 1
            num_batch += 1

    if len(elapsed_time_per_batch) > 0:
        print(f"Mean batch computing time: {sum(elapsed_time_per_batch)/len(elapsed_time_per_batch)} seconds")
        print(f"Batches times: {elapsed_time_per_batch}")
    else:
        print("No times for batches!")
if __name__ == "__main__":
    main()