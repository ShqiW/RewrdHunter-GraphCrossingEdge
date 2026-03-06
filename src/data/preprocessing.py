from torch_geometric.data import InMemoryDataset, Data

class RomeDataset(InMemoryDataset):
    def __init__(self, root, split="train", transform=None):
        self.split = split
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']
    
    def process(self):
        data_list = []
        for graph_file in tqdm(self.raw_file_names):
            G = nx.read_graphml(graph_file)
            data = Data(
                edge_index=...,
                degree=...,
                graph_distance=...,
                P_graph=...,
            )
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])