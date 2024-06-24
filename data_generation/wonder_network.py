from datetime import timedelta
from typing import List

import requests


def get_ping_ids() -> List[int]:
    ids: str = '2,4,6,16,17,285,105,145,215,142,241,303,7,201,247,189,110,309,214,18,8,143,294,40,222,64,236,251,204,292,48,250,5,234,213,14,238,147,21,228,200,181,208,67,47,50,96,205,62,132,218,42,242,191,260,78,263,85,299,22,104,59,137,308,138,288,123,80,26,91,233,165,12,115,71,144,81,148,291,196,133,19,153,187,103,229,301,264,63,29,30,226,206,161,157,60,102,197,177,99,83,203,84,65,45,74,232,72,151,39,56,298,230,155,66,227,122,75,57,258,0,175,118,55,128,174,194,302,297,124,244,212,36,34,243,186,295,79,188,223,108,192,259,44,28,11,37,211,61,240,94,156,77,114,112,237,49,1,231,33,15,27,252,92,167,106,52,9,293,210,307,131,88,46,140,13,310,176,158,89,119,129,256,300,195,98,169,68,209,38,130,184,82,3,111,249,254,162,225,76,127,116,190,163,54,93,141,86,51,73,100,23,168,70,125,166,239,109,24,171,113,306,180,221,126,31,305,154,304,90,10,149,185,107,235,95,25,97,245,87,216,117,193,136,248,58,246,53,69,202,20,170,32,139,217,135,150,261,35'
    separted_ids: List[int] = [int(x) for x in ids.split(',')]
    separted_ids.pop(26)
    separted_ids.pop(47)
    separted_ids.pop(50)
    separted_ids.pop(91)
    return separted_ids


def get_random_latency_matrix_of_size(size: int):
    # random_ids = random.sample(get_ping_ids(), size + 1)
    random_ids = (get_ping_ids())[:size]
    random_ids_joined = ",".join(str(elem) for elem in random_ids)
    url = f"https://wondernetwork.com/ping-data?sources={random_ids_joined}&destinations={random_ids_joined}"
    response = requests.get(url).json()
    ping_data = response["pingData"]
    latency_matrix = []
    for latency_data_for_source in ping_data.values():
        latencies_from_source = []
        for latency in latency_data_for_source.values():
            latencies_from_source.append(timedelta(milliseconds=float(latency['avg'])))
        latency_matrix.append(latencies_from_source)

    return latency_matrix



