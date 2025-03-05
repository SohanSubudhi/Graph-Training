import numpy as np
import os
import pickle
import networkx as nx
import torch_geometric

from parse_logs import FIELDS_TO_ADD, NODE_TYPE_LIST


def connect_graph(G, unconnected_nodes, new_node_name):
    new_node = G.nodes[new_node_name]
    while True:
        if len(unconnected_nodes) == 0:
            return
        
        last_node = unconnected_nodes[-1]
        last_node_start = G.nodes[last_node]['start']
        last_node_end   = last_node_start + G.nodes[last_node]['duration']

        if new_node['start'] <= last_node_start and (new_node['start'] + new_node['duration']) >= last_node_end:
            G.add_edge(new_node_name, last_node)
            del unconnected_nodes[-1]
        else: 
            return

def make_graph(nodes, events, log_path=None, use_cache=True):
    if log_path is not None:
        assert os.path.exists(log_path)
        cache_path = log_path[:-4] + ".graphcache"

        if os.path.exists(cache_path):
            if use_cache:
                print(f"Using cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"use_cache=False. Deleting {cache_path} and recreating.")
                os.remove(cache_path)
    
    PID = nodes[nodes['event'] == 'sched_process_exit']['pid'].item()
    print(f"Found PID: {PID}")
    # CPU nodes
    wakeup_events = list(
        filter(
            lambda x: x['pid'] == PID,
            map(lambda x: x, nodes[(nodes['event'] == 'sched_wakeup_new')|(nodes['event'] == 'sched_wakeup')]['data'])
        )
    )
    assert len(wakeup_events) == 1, f"Have {len(wakeup_events)} events :-("
    
    switch_events = nodes[nodes['event'] == 'sched_switch']
    exit_events = nodes[nodes['event'] == 'sched_process_exit']

    START_TIME = wakeup_events[0]['time']
    END_TIME = exit_events['time'].item() - START_TIME

    # return wakeup_events, switch_events, exit_events 
    cpu_starts = []
    cpu_ends = []
    
    cpu_starts.append(wakeup_events[0]['time'])
    
    for row in switch_events.iterrows():
        if row[1].code == 1:
            cpu_ends.append(row[1].data['time'])
        else:
            cpu_starts.append(row[1].data['time'])

    cpu_ends.append(exit_events['time'].item())

    cpu_starts = np.asarray(cpu_starts) - START_TIME
    cpu_ends = np.asarray(cpu_ends) - START_TIME
    
    assert len(cpu_starts) == len(cpu_ends), "What starts, must end."

    # init Graph and add CPU nodes
    G = nx.DiGraph()
    G.add_node("S")
    
    G.nodes['S']['start'] = 0
    G.nodes['S']['duration'] = END_TIME
    G.nodes['S']['type'] = 'S'

    stack = dict()
    index = {}
    unconnected_nodes = []
    
    for row in nodes.iterrows():
        idx, row = row
        if row.event in ['sched_fork', 'sched_switch', 'sched_wakeup', 'sched_wakeup_new', 'sched_process_exit']:
            if 'cpu' not in index.keys():
                index['cpu'] = 0
            if 'cpu' not in stack.keys():
                stack['cpu'] = []
            
            if (row.event == 'sched_wakeup_new' and row.data['pid'] == PID) or (row.event == 'sched_switch' and row.code == 0):
                stack['cpu'].append([f"cpu#{index['cpu']}", row.data])
                index['cpu'] += 1
            
            elif row.event == 'sched_process_exit' or (row.event == 'sched_switch' and row.code == 1):
                assert len(stack['cpu']) > 0 
                last_elem = stack['cpu'][-1]
                del stack['cpu'][-1]
                G.add_node(last_elem[0])
                G.nodes[last_elem[0]]['type'] = 'cpu'
                G.nodes[last_elem[0]]['min_faults'] = []
                G.nodes[last_elem[0]]['maj_faults'] = []
                G.nodes[last_elem[0]]['start'] = last_elem[1]['time'] - START_TIME
                G.nodes[last_elem[0]]['duration'] = row.time - last_elem[1]['time']
                
                connect_graph(G, unconnected_nodes,  last_elem[0])
                unconnected_nodes.append(last_elem[0])
                
                del last_elem
        
        elif row.event == 'blk_account':
            if row.event not in index.keys():
                index[row.event] = 0
            
            if row.event not in stack.keys():
                stack[row.event] = {}

            if row.code == 0:
                assert row.data['ident'] not in stack[row.event].keys(), f"Remove assertion if it gets raised - but do a soft check later: {row}"
                stack[row.event][row.data['ident']] = [f"blk_account#{index[row.event]}", row.data]
                index[row.event] += 1
            else:
                if row.data['ident'] not in stack[row.event].keys():
                    print(f"Missing blk event: {row.data['ident']} not found")
                    continue 
                node_name, start_event_data = stack[row.event][row.data['ident']]
                del stack[row.event][row.data['ident']]
                
                attr_to_save = FIELDS_TO_ADD[row.event]
                
                kv_dict = {}
                for attr in attr_to_save:
                    if attr[1] in start_event_data.keys():
                        kv_dict[attr[0]] = start_event_data[attr[1]]
                    else:
                        kv_dict[attr[0]] = row.data[attr[1]]

                G.add_node(node_name, **kv_dict)
                G.nodes[node_name]['type'] = node_name.split('#')[0]
                G.nodes[node_name]['start'] = start_event_data['time'] - START_TIME
                G.nodes[node_name]['duration'] = row.time - start_event_data['time']

                connect_graph(G, unconnected_nodes,  node_name)
                unconnected_nodes.append(node_name)
                del node_name, start_event_data, attr_to_save
        else:
            if row.event not in index.keys():
                index[row.event] = 0
            
            if row.event not in stack.keys():
                stack[row.event] = []
            
            if row.code == 0:
                stack[row.event].append([f"{row.event}#{index[row.event]}", row.data])
                index[row.event] += 1
            else:
                assert len(stack[row.event]) > 0, f"event={row.event}, data={row.data}, idx={idx}"
                last_elem = stack[row.event][-1]
                del stack[row.event][-1]

                attr_to_save = FIELDS_TO_ADD[row.event]
                kv_dict = {}
                for attr in attr_to_save:
                    if attr[1] in last_elem[-1].keys():
                        kv_dict[attr[0]] = last_elem[-1][attr[1]]
                    else:
                        kv_dict[attr[0]] = row.data[attr[1]]

                G.add_node(last_elem[0], **kv_dict)
                G.nodes[last_elem[0]]['type'] = last_elem[0].split('#')[0]
                G.nodes[last_elem[0]]['start'] = last_elem[1]['time'] - START_TIME
                G.nodes[last_elem[0]]['duration'] = row.time - last_elem[1]['time']

                connect_graph(G, unconnected_nodes,  last_elem[0])
                unconnected_nodes.append(last_elem[0])
                del last_elem, attr_to_save, kv_dict

    # annotate CPU nodes with events
    for event in events:
        event_ts = event['timestamp'] - START_TIME
        assigned = False
        
        for i in range(0, index['cpu']):
            start_cpu = G.nodes[f"cpu#{i}"]['start']
            end_cpu = start_cpu + G.nodes[f"cpu#{i}"]['duration']
            if event_ts >= start_cpu and event_ts <= end_cpu:
                for j in range(event['value']):
                    G.nodes[f"cpu#{i}"][event['name']].append(event_ts)
                assigned = True
                break
        
        if not assigned:
            print(f"Event {event} not within any CPU event - how?")

    # aggregate events - for now we are dount just COUNT, but can add more later.
    for i in range(0, index['cpu']):
        G.nodes[f'cpu#{i}']['maj_faults'] = len(G.nodes[f'cpu#{i}']['maj_faults'])
        G.nodes[f'cpu#{i}']['min_faults'] = len(G.nodes[f'cpu#{i}']['min_faults'])

    connect_graph(G, unconnected_nodes, 'S')
    assert len(unconnected_nodes) == 0

    if log_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump((G, stack), f)
    
    return G, stack

def convert_to_pyg_homogenous(G):
    ATTRIBUTES_TO_REMAIN = ['type', 'start', 'duration', 'bytes', 'min_faults', 'max_faults', 'cmd_flags', 'rq_flags']
    for node in G.nodes:
        try:
            type_idx, _ = NODE_TYPE_LIST[G.nodes[node]['type']]
        except:
            print(node)
        G.nodes[node]['type'] = [0] * len(NODE_TYPE_LIST)
        G.nodes[node]['type'][type_idx] = 1
        
    
        node_attr = list(G.nodes[node].keys())
        for attr in node_attr: 
            if attr not in ATTRIBUTES_TO_REMAIN:
                del G.nodes[node][attr]
    
        for attr in ATTRIBUTES_TO_REMAIN:
            if attr not in G.nodes[node].keys():
                G.nodes[node][attr] = 0
    return torch_geometric.utils.from_networkx(G, group_node_attrs=ATTRIBUTES_TO_REMAIN)