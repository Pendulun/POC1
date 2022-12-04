import pandas as pd
import osmnx as ox
import networkx as nx
import pathlib
import multiprocessing
import warnings

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

def get_indicators_df():
    indicators_from_paper_path = pathlib.Path('./data/indicators.csv')
    if not indicators_from_paper_path.exists():
        print(f"{indicators_from_paper_path} does not exists!")
        exit(-1)
    
    indicators_df = pd.read_csv(indicators_from_paper_path)
    return indicators_df

def get_graphml_files(graph_folder_path) -> list:
    return list(graph_folder_path.glob('*.graphml'))[:3]

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

def merge_all_features(features_dict_list:list) -> dict:
    full_features = features_dict_list[0]
    for features in features_dict_list[1:]:
        full_features = {**full_features, **features}
    
    return full_features

def compute_networkx_features(G) -> dict:
    features = dict()
    undirected_graph = nx.Graph(G)
    greatest_undirected_component_nodes = max(nx.connected_components(undirected_graph), key=len)
    # greatest_directed_component_graph = G.subgraph(greatest_undirected_component_nodes)
    greatest_undirected_component_graph = undirected_graph.subgraph(greatest_undirected_component_nodes)

    #Too expensive to compute
    # features['node_conectivity'] = len(nx.minimum_node_cut(greatest_component))
    # features['edge_conectivity'] = len(nx.minimum_edge_cut(greatest_component))
    # features['avg_conectivity'] = nx.average_node_connectivity(undirected_graph)
    
    betweenness_cent:dict = nx.betweenness_centrality(G)
    features['max_betweenness_centrality'] = max(list(betweenness_cent.values()))
    features['avg_betweenness_centrality'] = features['max_betweenness_centrality']/len(betweenness_cent)
    
    features['global_efficiency'] = nx.global_efficiency(undirected_graph)

    information_cent = nx.information_centrality(greatest_undirected_component_graph)
    features['max_info_centrality'] = max(list(information_cent.values()))
    features['avg_info_centrality'] = features['max_info_centrality']/len(information_cent)

    #This method already treats for multiple components.
    #See it's documentation for the Wasserman and Faust version
    closeness_cent = nx.closeness_centrality(G)
    features['avg_closeness_centrality'] = max(list(closeness_cent.values()))/len(closeness_cent)
    return features

def compute_features(graph_path:pathlib.Path, indicators_df:pd.DataFrame) -> dict:
    full_features = dict()
    full_features = complete_with_basic_info(graph_path, full_features)
    try:
        G = ox.load_graphml(graph_path)

        #Falta conseguir a área em metros quadrados para computar algumas outras features
        #Para isso, devo ler também o arquivo de features já existentes.
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

def clean_features(features):
    cleaned_features = list()
    for feature in features:
        cleaned_features.append(remove_features_already_computed(feature))
    
    return cleaned_features

def save_features_df(features_df):
    new_features_folder_path = pathlib.Path('./new_features')
    new_features_folder_path.mkdir(exist_ok=True)

    features_df.to_csv(new_features_folder_path / 'new_features.csv', index=False)

def main():
    graph_folder_path = pathlib.Path('C:\\Users\\User\\Documents\\UFMG\\POC1\\world_graphs\\')
    check_valid_folder(graph_folder_path)
    
    indicators_df = get_indicators_df()
    graph_files = get_graphml_files(graph_folder_path)
    
    features = [compute_features(graph_path, indicators_df) for graph_path in graph_files]
    features = clean_features(features)

    features_df = pd.DataFrame.from_records(features)
    features_df.fillna(0, inplace=True)

    save_features_df(features_df)

def check_valid_folder(graph_folder_path):
    if not graph_folder_path.exists():
        print(f'{graph_folder_path} does not exists!')
        exit(-1)
    
    if not graph_folder_path.is_dir():
        print(f"{graph_folder_path} is not a folder!")
        exit(-1)

if __name__ == "__main__":
    #So to ignore networkx warnings
    warnings.filterwarnings("ignore")
    main()