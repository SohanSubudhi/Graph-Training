from collections import defaultdict
from typing import Any, Iterable, List, Literal, Optional, Union

import torch
import torch_geometric


def from_hetero_networkx(
    G: Any, node_type_attribute: str,
    edge_type_attribute: Optional[str] = None,
    graph_attrs: Optional[Iterable[str]] = None, nodes: Optional[List] = None,
    group_node_attrs: Optional[Union[List[str], Literal['all']]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal['all']]] = None
) -> 'torch_geometric.data.HeteroData':
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.HeteroData` instance.
    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        node_type_attribute (str): The attribute containing the type of a
            node. For the resulting structure to be valid, this attribute
            must be set for every node in the graph. Values contained in
            this attribute will be casted as :obj:`string` if possible. If
            not, the function will raise an error.
        edge_type_attribute (str, optional): The attribute containing the
            type of an edge. If set to :obj:`None`, the value :obj:`"to"`
            will be used in the final structure. Otherwise, this attribute
            must be set for every edge in the graph. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        nodes (list, optional): The list of nodes whose attributes are to
            be collected. If set to :obj:`None`, all nodes of the graph
            will be included. (default: :obj:`None`)
        group_node_attrs (List[str] or "all", optional): The node attributes to
            be concatenated and added to :obj:`data.x`. They must be present
            for all nodes of each type. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`. They must be
            present for all edge of each type. (default: :obj:`None`)
    Example:
        >>> data = from_hetero_networkx(G, node_type_attribute="type",
        ...                    edge_type_attribute="type")
        <torch_geometric.data.HeteroData()>
    :rtype: :class:`torch_geometric.data.HeteroData`
    """
    import networkx as nx

    from torch_geometric.data import HeteroData

    def get_edge_attributes(G: Any, edge_indexes: list,
                            edge_attrs: Optional[Iterable] = None) -> dict:
        r"""Collects the attributes of a list of graph edges in a dictionary.
        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            edge_indexes (list, optional): The list of edge indexes whose
                attributes are to be collected. If set to :obj:`None`, all
                edges of the graph will be included. (default: :obj:`None`)
            edge_attrs (iterable, optional): The list of expected attributes to
                be found in every edge. If set to :obj:`None`, the first
                edge encountered will set the values for the rest of the
                process. (default: :obj:`None`)
        Raises:
            ValueError: If some of the edges do not share the same list
            of attributes as the rest, an error will be raised.
        """
        data = defaultdict(list)
        edge_to_data = list(G.edges(data=True))

        for edge_index in edge_indexes:
            _, _, feat_dict = edge_to_data[edge_index]
            if edge_attrs is None:
                edge_attrs = feat_dict.keys()
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes.')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        return data

    def get_node_attributes(
            G: Any, nodes: list,
            expected_node_attrs: Optional[Iterable] = None) -> dict:
        r"""Collects the attributes of a list of graph nodes in a dictionary.
        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            nodes (list, optional): The list of nodes whose attributes are to
                be collected. If set to :obj:`None`, all nodes of the graph
                will be included. (default: :obj:`None`)
            expected_node_attrs (iterable, optional): The list of expected
                attributes to be found in every node. If set to :obj:`None`,
                the first node encountered will set the values for the rest
                of the process. (default: :obj:`None`)
        Raises:
            ValueError: If some of the nodes do not share the same
            list of attributes as the rest, an error will be raised.
        """
        data = defaultdict(list)

        node_to_data = G.nodes(data=True)

        for node in nodes:
            feat_dict = node_to_data[node]
            if expected_node_attrs is None:
                expected_node_attrs = feat_dict.keys()
            if set(feat_dict.keys()) != set(expected_node_attrs):
                raise ValueError('Not all nodes contain the same attributes.')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        return data

    G = G.to_directed() if not nx.is_directed(G) else G

    if nodes is not None:
        G = nx.subgraph(G, nodes)

    hetero_data_dict = {}

    node_to_group_id = {}
    num_nodes = {}
    node_to_group = {}
    group_to_nodes = defaultdict(list)
    group_to_edges = defaultdict(list)

    for node, node_data in G.nodes(data=True):
        if node_type_attribute not in node_data:
            raise KeyError(f"Given node_type_attribute: {node_type_attribute} \
                missing from node {node}.")
        node_type = str(node_data[node_type_attribute])
        group_to_nodes[node_type].append(node)
        num_nodes[node_type] = num_nodes.get(node_type, 0) + 1
        node_to_group_id[node] = len(group_to_nodes[node_type]) - 1
        node_to_group[node] = node_type

    for i, (node_a, node_b, edge_data) in enumerate(G.edges(data=True)):
        if edge_type_attribute is not None:
            if edge_type_attribute not in edge_data:
                raise KeyError(
                    f"Given edge_type_attribute: {edge_type_attribute} \
                    missing from edge {(node_a, node_b)}.")
            node_type_a, edge_type, node_type_b = edge_data[
                edge_type_attribute]
            if node_to_group[node_a] != node_type_a or node_to_group[
                    node_b] != node_type_b:
                raise ValueError(f'Edge {node_a}-{node_b} of type\
                         {edge_data[edge_type_attribute]} joins nodes of types\
                         {node_to_group[node_a]} and {node_to_group[node_b]}.')
        else:
            edge_type = "to"
        group_to_edges[(node_to_group[node_a], edge_type,
                        node_to_group[node_b])].append(i)

    for group, group_nodes in group_to_nodes.items():
        hetero_data_dict[str(group)] = {
            k: v
            for k, v in get_node_attributes(G, nodes=group_nodes).items()
            if k != node_type_attribute
        }
        hetero_data_dict[str(group)]["num_nodes"] = [num_nodes[group]]
        hetero_data_dict[str(group)]["node_type"] = [str(group)]

    for edge_group, group_edges in group_to_edges.items():
        group_name = '__'.join(edge_group)
        hetero_data_dict[group_name] = {
            k: v
            for k, v in get_edge_attributes(G,
                                            edge_indexes=group_edges).items()
            if k != edge_type_attribute
        }
        edge_list = list(G.edges(data=False))
        global_edge_index = [edge_list[edge] for edge in group_edges]
        group_edge_index = [(node_to_group_id[node_a],
                             node_to_group_id[node_b])
                            for node_a, node_b in global_edge_index]
        hetero_data_dict[group_name]["edge_index"] = torch.tensor(
            group_edge_index, dtype=torch.long).t().contiguous().view(2, -1)
        
        graph_items = G.graph
    if graph_attrs is not None:
        graph_items = {
            k: v
            for k, v in graph_items.items() if k in graph_attrs
        }
    for key, value in graph_items.items():
        hetero_data_dict[str(key)] = value

    for group, group_dict in hetero_data_dict.items():
        if isinstance(group_dict, dict):
            xs = []
            is_edge_group = group in [
                '__'.join(k) for k in group_to_edges.keys()
            ]
            if is_edge_group:
                group_attrs = group_edge_attrs
            else:
                group_attrs = group_node_attrs
            for key, value in group_dict.items():
                if isinstance(value, (tuple, list)) and isinstance(
                        value[0], torch.Tensor):
                    hetero_data_dict[group][key] = torch.stack(value, dim=0)
                else:
                    try:
                        hetero_data_dict[group][key] = torch.tensor(value)
                    except (ValueError, TypeError):
                        pass
                if group_attrs is not None and key in group_attrs:
                    xs.append(hetero_data_dict[group][key].view(-1, 1))
            if group_attrs is not None:
                if len(group_attrs) != len(xs):
                    raise KeyError(
                        f'Missing required attribute in group: {group}')
                if is_edge_group:
                    hetero_data_dict[group]['edge_attr'] = torch.cat(
                        xs, dim=-1)
                else:
                    hetero_data_dict[group]['x'] = torch.cat(xs, dim=-1)
        else:
            value = group_dict
            if isinstance(value, (tuple, list)) and isinstance(
                    value[0], torch.Tensor):
                hetero_data_dict[group] = torch.stack(value, dim=0)
            else:
                try:
                    hetero_data_dict[group] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass
    
    graph = HeteroData(**hetero_data_dict)
    print(graph.node_types)
    print(graph.edge_types)
    for nodeStorage in graph.node_stores:
        print(nodeStorage.num_nodes)
    return graph