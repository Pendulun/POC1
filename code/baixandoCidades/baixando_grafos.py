import osmnx as ox
import pathlib
import matplotlib.pyplot as plt
import threading
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor as Executor
ox.__version__

class GraphSaver():
    def __init__(self, graph_folder_path:str):
        self._error_lock = threading.Lock()
        self.error_list = list()
        
        self.graph_folder = pathlib.Path(graph_folder_path)
        self.graph_folder.mkdir(parents=True, exist_ok=True)
        
        self._stats_lock = threading.Lock()
        self.graph_stats = dict()
        
        self._download_counter_lock = threading.Lock()
        self._download_counter = 0
    
    def download_graph_from_places(self, places_names_and_ids:list, n_threads:int = None, verbose:bool = True, print_every:int = 10):
        self.error_list.clear()
        with Executor(max_workers=n_threads) as executor:
            [executor.submit(self._download_and_save_graph_from_place, place_id, place_name, verbose, print_every) for place_id, place_name in places_names_and_ids]
    
    def _download_and_save_graph_from_place(self, place_id:int, place_name:str, verbose:bool, print_every:int):
        
        if verbose:
            with self._download_counter_lock:
                if self._download_counter % print_every == 0:
                    print(f"({place_id}, {place_name})")

                self._download_counter += 1
            
        try:
            G = ox.graph_from_place(place_name, network_type="drive")
        except:
            with self._error_lock:
                self.error_list.append((place_id, place_name))    
            return
        
        graph_file = self.graph_folder / f'{place_id}.graphml'
        ox.io.save_graphml(G, graph_file)

        n_nodes = len(G.nodes)
        n_edges = len(G.edges)
        self._save_graph_stats(place_id, n_nodes, n_edges)
    
    def _save_graph_stats(self, place_id:int, n_nodes:int, n_edges:int):
        with self._stats_lock:
            place_stats = self.graph_stats.setdefault(place_id, dict())
            place_stats['n_nodes'] = n_nodes
            place_stats['n_edges'] = n_edges
    
    def save_graph_plot(self, G, graph_id:int, img_folder_path:str):
        n_nodes = len(G.nodes)
        node_size = 10
        
        if n_nodes > 10000:
            node_size = 0
        elif n_nodes > 5000:
            node_size = 1
        elif n_nodes > 2000:
            node_size = 2
            
        show=False
        close=True
        save=True
        filepath= img_folder_path / f'{graph_id}.png'
        
        fig, ax = ox.plot_graph(G, node_size=node_size, show=show, close=close, save=save, filepath=filepath)
    
    def save_error_list_to_file(self, path:str):
        error_df = pd.DataFrame(self.error_list, columns=['place_id', 'place_name'])
        error_df.sort_values(by="place_id", inplace=True)
        error_df.set_index("place_id")

        error_list_path = pathlib.Path(path)
        error_df.to_csv(error_list_path, sep=';', index=False)
    
    def save_stats_to_file(self, path:str):
        stats_df = pd.DataFrame(columns=['place_id', 'n_nodes', 'n_edges'])
        for place_id, stats in self.graph_stats.items():
            stats_df.loc[len(stats_df.index)] = [place_id, stats['n_nodes'], stats['n_edges']]
        
        stats_df.sort_values(by="place_id", inplace=True)
        stats_df.set_index("place_id")

        stats_file_path = pathlib.Path(path)
        stats_df.to_csv(stats_file_path, index=False)

def config_args_parser():
    new_arg_parser = argparse.ArgumentParser()
    new_arg_parser.add_argument("--n_places", type=int, required=True,
                        help="The max number of places to download")
                        
    return new_arg_parser

        
if __name__ == "__main__":
    arg_parser = config_args_parser()
    user_args = arg_parser.parse_args()

    places_df = pd.read_csv('data/final_strings.csv')['0']
    target_places = list(places_df.items())[:user_args.n_places]

    graphs_folder_path = "C:\\Users\\User\\Documents\\UFMG\\POC1\\graphs"
    gd = GraphSaver(graphs_folder_path)
    gd.download_graph_from_places(target_places, print_every=5)

    gd.save_error_list_to_file("error_list.csv")

    gd.save_stats_to_file("stats.csv")