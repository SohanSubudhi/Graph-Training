import json
import os
import pandas as pd
import pickle
import re

KEY_FORMAT = {
    'min_faults': ['pid', 'timestamp'],
    'maj_faults': ['pid', 'timestamp'],
    'started': ['addr']
}

def process_associative_array(x):
    name, other_data = x.split('[')
    name = name.replace('@', '')
    other_data = other_data.replace(']', '')
    key, value = other_data.split(':')
    key = key.strip().split(', ')
    assert len(key) == len(KEY_FORMAT[name]), f"Check KEY_FORMAT for {name}"
    value = value.strip()
    try:
        value = int(value)
    except: 
        pass
    return_dict = {
        'name': name, 
        'value': value
    }
    for i in range(len(KEY_FORMAT[name])):
        try:
            key[i] = int(key[i])
        except:
            pass
        return_dict[KEY_FORMAT[name][i]] = key[i]    
    return return_dict


def process_json(x, t = None):
    if len(x) < 5:
        data = ''
    else:
        data = '{' + x[4]
    
    data = data.strip()
    return_dict = {}
    
    try:
        if data is not None and data.startswith('{'):
            return_dict = json.loads(data.strip().replace("'", '"'))
    except:
        print("<", data, ">")
        raise NotImplementedError()
    
    if t is not None:
        return_dict['time'] = t
    
    return return_dict

EVENT_MAPPING = {
    999: "sched_fork",
    1: "sched_wakeup",
    2: "sched_wakeup_new",
    3: "sched_switch",
    4: "ksys_read",
    5: "ksys_write",
    7: "sys_openat",
    8: "sys_open",
    9: "sys_close",
    10: "vfs_read",
    11: "new_sync_read",
    # 12: "ext4_file_read_iter",
    # 13: "generic_file_read_iter",
    # 14: "filemap_read",
    # 15: "filemap_get_pages",
    17: "ext4_mpage_readpages",
    18: "ext4_readahead",
    19: "read_pages", 
    20: "page_cache_ra_unbounded",
    21: "ondemand_readahead",
    22: "page_cache_sync_ra",
    23: "page_cache_async_ra",
    # 24: "filemap_get_read_batch",
    26: "blk_account",
    -1: "sched_process_exit"
}

REVERSE_EVENT_MAPPING = {}
for k in EVENT_MAPPING:
    REVERSE_EVENT_MAPPING[EVENT_MAPPING[k]] = k 

FIELDS_TO_ADD = {
    "sched_fork": [],
    "sched_wakeup": [],
    "sched_wakeup_new": [],
    "sched_switch": [],
    "ksys_read": [['bytes', 'count']],
    "ksys_write": [['bytes', 'count']],
    "sys_openat": [],
    "sys_open": [],
    "sys_close": [],
    "vfs_read": [['bytes', 'count']],
    "new_sync_read": [['bytes', 'len']],
    # "ext4_file_read_iter": [['retval', 'retval']],
    # "generic_file_read_iter": [['retval', 'retval']],
    # "filemap_read": [['retval', 'retval']],
    # "filemap_get_pages": [],
    "ext4_mpage_readpages": [],
    "ext4_readahead": [],
    "read_pages": [], 
    "page_cache_ra_unbounded": [],
    "ondemand_readahead": [],
    "page_cache_sync_ra": [],
    "page_cache_async_ra": [],
    # "filemap_get_read_batch": [],
    "blk_account": [['bytes', 'len'], ['ident', 'ident'], ['cmd', 'cmd'], ['rq', 'rq']],
    "sched_process_exit": []
}

assert len(FIELDS_TO_ADD) == len(EVENT_MAPPING)

'''
    In the 2-tuples below:
        -> the first element of the tuple represents the one hot encoding of this type of node.
        -> the second element in the tuple represents the node type, when using heterogenous node
                -> Type 0 Node: start, duration
                -> Type 1 Node: start, duration, cmd flags, rq flags (used for block IO calls only)
                -> Type 2 Node: start, duration, min faults, maj faults (for CPU nodes only)
                -> Type 3 Node: start, duration, bytes (any kfunc involving memory/data)  
'''

NODE_TYPE_LIST = {
    'S': (0, 0), 
    'blk_account': (1, 1), 
    'cpu': (2, 2),
    'ext4_mpage_readpages': (3, 0), 
    'ext4_readahead': (4, 0),
    'ksys_read': (5, 3),
    'ksys_write': (6, 3),
    'new_sync_read': (7, 3),
    'ondemand_readahead': (8, 0),
    'page_cache_async_ra': (9, 0),
    'page_cache_ra_unbounded': (10, 0),
    'page_cache_sync_ra': (11, 0),
    'read_pages': (12, 0),
    'sys_close': (13, 0),
    'sys_openat': (14, 0),
    'vfs_read': (15, 3)
}


def parse_logs(file, all_data=None, use_cache=True):
    assert file.endswith(".log")
    cache_path = file[:-4] + ".cache"

    if os.path.exists(cache_path):
        if use_cache:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"use_cache=False. Deleting {cache_path} and recreating.")
            os.remove(cache_path)
    
    if all_data is None:
        with open(file, 'r') as f:
            all_data = f.readlines()
    
    nodes = map(
        lambda x: re.split(r'\]\s*\[|\]\s*\{', x), 
        filter(lambda y: y.startswith('['), all_data)
    )

    nodes = map(
        lambda x: {
            'time': int(x[0].replace('[', '').replace(']', '')),
            'pid':  int(x[1].replace('[', '').replace(']', '')),
            'event': EVENT_MAPPING[int(x[2].replace('[', '').replace(']', ''))],
            'code': int(x[3].replace('[', '').replace(']', '')),
            'data': process_json(x, t = int(x[0].replace('[', '').replace(']', '')))
        },
        nodes
    )
    
    nodes = pd.DataFrame(nodes)
    
    events = list(
        map(
            process_associative_array,
            filter(lambda x: x.startswith('@'), all_data)
        )
    )

    with open(cache_path, 'wb') as f:
        pickle.dump((nodes, events), f)

    return nodes, events