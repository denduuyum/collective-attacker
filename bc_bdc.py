"""
   Critical region сонгон аваад, Липтон Тражан ашиглан хэд хэд тайрна.
   Түүний дараагаар тайрсан оройнуудыг буцаан нэмнэ.
"""
import networkit as nk
from sys import stdin, stdout
import sys
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
import copy
import UnionFind as UF
import graph_tools
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

OUTPUT_DIR = "./output/Design1"

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    # print(*args, file=sys.stdout, **kwargs)

def s_score(G, nodes, uf):
    score = {}

    H = 0
    for sz in uf.sizes():
        H += sz * (sz - 1) / 2
    
    for u in nodes:
        t_test = [u]
        total = uf.size(u)
        sub = uf.size(u) * (uf.size(u) - 1) / 2
        for v in G.iterNeighbors(u):
            x = uf.find(v)
            if uf.same(u, x) == False and x not in t_test:
                t_test.append(x)
                total += uf.size(x)
                sub += uf.size(x) * (uf.size(x) - 1) / 2

        score[u] = H + total * (total - 1) / 2 - sub
    return score

    

def add_back(G, SG, n, cut_nodes, deleted_nodes = None, beta = None, limit = None, bc_data = None):
    # print('Nodes of SG: ', list(SG.iterNodes()))
    # print('Nodes of G: ', list(G.iterNodes()))
    nn = SG.numberOfNodes() + len(cut_nodes)
    
    # bc = nk.centrality.EstimateBetweenness(G, math.log2(n))
    # bc.run()
    # bc_data = bc.scores()

    # Get permutation of sorted cut nodes by betweenness
    # p = [i for i in range(len(cut_nodes))]
    # p = sorted(p, key = lambda i: bc_data[cut_nodes[i]])

    uf_bk = UF.UnionFind(n)
    C = 0                     # size of the biggest component
    for e in SG.iterEdges():
        uf_bk.union(e[0], e[1])
        if uf_bk.size(e[0]) > C:
            C = uf_bk.size(e[0])
            
    log_limit = 0
    start = C - 1
    if limit == None:
        if nn - C <= 0:
            log_limit = 0
        else:
            log_limit = math.floor(math.log2(nn - C))
    else:
        log_limit = 0
        start = limit - 1

    best_cut = [True] * len(cut_nodes)
    # Best cut damage
    # It's measured as |previous_biggest_component_size - current_size| / size_of_cuts
    best_H = 0
    for t in uf_bk.sizes():
        best_H += t * (t - 1) / 2
    best_H = best_H * len(cut_nodes) * len(cut_nodes)

    ssize = s_score(G, cut_nodes, uf_bk)
    # Get permutation of sorted cut nodes by their neighboring set sizes
    p = [i for i in range(len(cut_nodes))]
    p = sorted(p, key = lambda i: ssize[cut_nodes[i]])
    
    eprint('------ Searching the best B: the biggest component size is ', C)

    R = math.sqrt(n)
    if beta != None:
        R *= beta

    for j in range(0, log_limit + 1):
        Ci = start + 2**j
        uf = copy.copy(uf_bk)
        maxC = C
        count_added_back_nodes = 0
        is_del = [True] * len(cut_nodes)
        added_nodes = []

        eprint('Trying B*L = ', Ci, end = ' ')
    
        for i in range(len(cut_nodes)):
            
            if limit == None and len(cut_nodes) - count_added_back_nodes <=  R:
                break

            u = cut_nodes[p[i]]
            t = uf.size(u)
            t_test = []
            for v in G.iterNeighbors(u):
                x = uf.find(v)

                if uf.same(u, x) == False and x not in t_test and SG.hasNode(v):
                    t += uf.size(x)
                    t_test.append(x)
            
            if t <= Ci:
                is_del[p[i]] = False
                count_added_back_nodes += 1

                for v in t_test:
                    uf.union(u, v)
                    
                added_nodes.append(u)
                SG.restoreNode(u)

        for u in added_nodes:
            SG.removeNode(u)

        eprint('Number of added back nodes: ', len(added_nodes), end = ' ')

        H = 0
        for t in uf.sizes():
            H += t * (t - 1) / 2

        k = len(cut_nodes) - count_added_back_nodes
        H = H * k

        eprint('k = ', k, end = ' ')        
        eprint('Damage = ', H)
        if best_H > H and H > 0:
            best_cut = copy.copy(is_del)
            best_H = H

    dnodes = []
    for i in range(len(cut_nodes)):
        if best_cut[i]:
            dnodes.append(cut_nodes[i])
    if limit:
        graph_tools.draw_graph(nk.graphtools.subgraphFromNodes(G, cut_nodes), dnodes)
    # graph_tools.draw_highlight_top_bc(nk.graphtools.subgraphFromNodes(G, cut_nodes), bc_data)
    # nodes = [u for u in G.iterNodes()]
    # for i in range(len(nodes)):
    #     j = rd.randint(0, len(nodes) - 1)
    #     nodes[i], nodes[j] = nodes[j], nodes[i]
        
    # for i in range(int(math.sqrt(n))):
    #     dnodes.append(nk.graphtools.randomNode(G))
    # print(len(nodes))
    # graph_tools.draw_highlight_top_bc(nk.graphtools.subgraphFromNodes(G, nodes[:int(n/2)]), bc_data)
    dlen = 0
    for i in range(len(cut_nodes)):
        if best_cut[i]:
            G.removeNode(cut_nodes[i])
            dlen += 1
            if deleted_nodes != None:
                deleted_nodes.append(cut_nodes[i])


    eprint('--------- After greedy: ')
    eprint('Number of deletion (number_of_actual_deleted_nodes / number_of_selected_nodes_for_deletion): ', dlen, '/', len(cut_nodes))
    #graph_tools.print_graph(G)

    
def find_bridge_nodes(G, n, bc_data):
    rnodes = []
    lnodes = []
    p = [u for u in G.iterNodes()]
    p = sorted(p, key = lambda u : bc_data[u], reverse = True)
    for i in range(0, int(math.sqrt(n))):
        if i >= len(p):
            break
        rnodes.append(p[i])

    return lnodes, rnodes

def delete_highest_degree(G, k):
    p = [u for u in G.iterNodes()]
    p = sorted(p, key = lambda u : G.degree(u), reverse = True)
    dnodes = p[:k]
    graph_tools.draw_graph(nk.graphtools.subgraphFromNodes(G, dnodes), dnodes)
    graph_tools.remove_nodes(G, dnodes)
    return dnodes

def bc_greedy_bc(G, n, nlog, reinsertion, beta, limit):
    BG = nk.Graph(G)

    deleted_nodes = []
    bc_iter = 0
    cz_alpha = 0
    while True:
        eprint()
        eprint('Iteration: ', bc_iter + 1, ' -------------------------------------------------------')
        bc_iter += 1
        eprint('Number of deleted nodes:', len(deleted_nodes))

        SG = nk.components.ConnectedComponents.extractLargestConnectedComponent(G)

        nn, mm = nk.graphtools.size(SG)
        eprint('Largest component size: ', nn, mm)
        
        if nn <= limit:
            break

        logn_pow = 1
        for i in range(nlog):
            logn_pow *= math.log2(nn)
        logn_pow = int(logn_pow)
        bc = nk.centrality.EstimateBetweenness(SG, logn_pow)
        bc.run()

        bc_data = bc.scores()

        lnodes, rnodes = find_bridge_nodes(SG, n, bc_data)
        eprint('lnodes, rnodes: ', len(lnodes), len(rnodes))
        cut_nodes = []
        if len(rnodes) == 0:
            cz_alpha += 1
            maxv = nk.graphtools.randomNode(SG)
            max_bc = bc_data[maxv]
            for u in SG.iterNodes():
                if max_bc < bc_data[u]:
                    max_bc = bc_data[u]
                    maxv = u
            cut_nodes += [maxv]
            SG.removeNode(maxv)
        else:
            cz_alpha = 0

        if cz_alpha >= 200:
            sys.exit(-1)

        graph_tools.remove_nodes(SG, rnodes)

        cut_nodes += rnodes
            
        eprint('Components in the biggest component: ')
        # graph_tools.print_graph(SG)

        if reinsertion:
            add_back(G, SG, n, cut_nodes, deleted_nodes, beta = beta, bc_data = bc_data)
        else:
            for v in cut_nodes[:int(math.sqrt(n)*beta)]:
                if G.hasNode(v):
                    G.removeNode(v)
                    deleted_nodes.append(v)


    IG = nk.Graph(BG)
    final_deleted_nodes = []
    eprint('Before final add back: ')
    # graph_tools.print_graph(G)
    add_back(BG, G, n, deleted_nodes, final_deleted_nodes, beta = beta, limit = limit)

    for u in final_deleted_nodes:
        IG.removeNode(u)
    return final_deleted_nodes

def bc_greedy_bdc(G, n, nlog, reinsertion, beta, limit):
    BG = nk.Graph(G)

    deleted_nodes = []
    bc_iter = 0
    cz_alpha = 0
    while True:
        eprint()
        eprint('Iteration: ', bc_iter + 1, ' -------------------------------------------------------')
        bc_iter += 1
        eprint('Number of deleted nodes:', len(deleted_nodes))

        SG = nk.components.ConnectedComponents.extractLargestConnectedComponent(G)

        nn, mm = nk.graphtools.size(SG)
        eprint('Largest component size: ', nn, mm)
        
        if nn <= limit:
            break

        logn_pow = 1
        for i in range(nlog):
            logn_pow *= math.log2(nn)
        logn_pow = int(logn_pow)
        bc = nk.centrality.EstimateBetweenness(SG, logn_pow)
        bc.run()

        bc_data = bc.scores()
        for u in SG.iterNodes():
            if SG.degree(u) == 0:
                bc_data[u] = 0
            else:
                bc_data[u] /= SG.degree(u)

        lnodes, rnodes = find_bridge_nodes(SG, n, bc_data)
        eprint('lnodes, rnodes: ', len(lnodes), len(rnodes))
        cut_nodes = []
        if len(rnodes) == 0:
            cz_alpha += 1
            maxv = nk.graphtools.randomNode(SG)
            max_bc = bc_data[maxv]
            for u in SG.iterNodes():
                if max_bc < bc_data[u]:
                    max_bc = bc_data[u]
                    maxv = u
            cut_nodes += [maxv]
            SG.removeNode(maxv)
        else:
            cz_alpha = 0

        if cz_alpha >= 200:
            sys.exit(-1)

        graph_tools.remove_nodes(SG, rnodes)

        cut_nodes += rnodes
            
        eprint('Components in the biggest component: ')
        # graph_tools.print_graph(SG)

        if reinsertion:
            add_back(G, SG, n, cut_nodes, deleted_nodes, beta = beta)
        else:
            for v in cut_nodes[:int(math.sqrt(n) * beta)]:
                if G.hasNode(v):
                    G.removeNode(v)
                    deleted_nodes.append(v)

    IG = nk.Graph(BG)
    final_deleted_nodes = []
    eprint('Before final add back: ')
    # graph_tools.print_graph(G)
    add_back(BG, G, n, deleted_nodes, final_deleted_nodes, beta = beta, limit = limit)

    for u in final_deleted_nodes:
        IG.removeNode(u)
    return final_deleted_nodes

def highest_degree(G, n, nlog, reinsertion, beta, limit):
    BG = nk.Graph(G)

    deleted_nodes = []
    bc_iter = 0
    cz_alpha = 0
    while True:
        eprint()
        eprint('Iteration: ', bc_iter + 1, ' -------------------------------------------------------')
        bc_iter += 1
        eprint('Number of deleted nodes:', len(deleted_nodes))

        SG = nk.components.ConnectedComponents.extractLargestConnectedComponent(G)

        nn, mm = nk.graphtools.size(SG)
        eprint('Largest component size: ', nn, mm)
        
        if nn <= limit:
            break

        rnodes = delete_highest_degree(SG, int(math.sqrt(n)))
        graph_tools.remove_nodes(G, rnodes)
        deleted_nodes += rnodes
        
            
    IG = nk.Graph(BG)
    final_deleted_nodes = []
    eprint('Before final add back: ')
    # graph_tools.print_graph(G)
    add_back(BG, G, n, deleted_nodes, final_deleted_nodes, beta = beta, limit = limit)

    for u in final_deleted_nodes:
        IG.removeNode(u)
    return final_deleted_nodes


def bc_greedy(G, n, nlog, reinsertion, beta, limit):

    s_clock = time.perf_counter()
    v1 = highest_degree(nk.Graph(G), n, nlog, reinsertion, beta, limit)
    # v1 = bc_greedy_bc(nk.Graph(G), n, nlog, reinsertion, beta, limit)
    # v2 = bc_greedy_bdc(nk.Graph(G), n, nlog, reinsertion, beta, limit)
    e_clock = time.perf_counter()
    # graph_tools.print_graph(IG)
    return v1, v2, e_clock - s_clock

def arg_setup():
    parser = argparse.ArgumentParser(description='Betweenness separator.')
    parser.add_argument('-b', '--beta', dest='beta', default = 1, type = float,
                        help='Minimum number of deleted vertices to be kept in reinsertion (0 < b <= 1)')
    parser.add_argument('--reinsertion', dest='reinsertion', action='store_true')
    parser.add_argument('--no-reinsertion', dest='reinsertion', action='store_false')
    parser.set_defaults(reinsertion = False)
    parser.add_argument('-n', '--nrun', dest='number', type = int, help='The number of runs', default = 10)
    parser.add_argument('-L', '--limit', dest='limit', type = int, required = True,
                    help='The biggest component size')
    parser.add_argument('-l', '--log', dest='log', type = int, default = 1,
                    help='Number of log(n) power in EstimateBetweenness')

    parser.add_argument('file', help = 'The file that contains graph information')
    return parser

def main(): 
    # Let's run in 4 threads
    nk.setNumberOfThreads(1)

    parser = arg_setup()
    args = parser.parse_args()

    with open(args.file, 'r') as fin:
        G, n = graph_tools.read_graph_mapped(fin)
        
    G = graph_tools.clean_graph(G)
    
    BG = nk.Graph(G)
    mx_cut = None
    mx_v = n
    mx_time = -1
    res_future = []
    res = []
    out = []
    is_bc = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(args.number):
            future = executor.submit(bc_greedy, nk.Graph(G), n, args.log, args.reinsertion, beta = args.beta, limit = args.limit)
            res_future.append(future)

    for i, future in enumerate(res_future):
        cut_bc, cut_bdc, duration = future.result()
        out.append([i, len(cut_bc), len(cut_bdc), duration])
        # print(len(cut), duration)
        cmin = len(cut_bc)
        ccut = cut_bc
        cbc = True
        if cmin > len(cut_bdc):
            cmin = len(cut_bdc)
            ccut = cut_bdc
            cbc = False
        if mx_v > cmin:
            mx_v = cmin
            mx_cut = ccut
            mx_time = duration
            is_bc = cbc

    fname = 'D1_result_'
    if args.reinsertion:
        fname += ''
    fname += str(args.beta) + '_' + str(args.number) + '_' + str(args.log) + '_'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = OUTPUT_DIR + '/' + fname + args.file.split('/')[-1]  + '.out'
    with open(fname, 'w') as f:
        f.write('# Test data: ' + args.file + '\n')
        f.write('# Parameters: ' + 'L = ' + str(args.limit) + ', beta = ' + str(args.beta) +
                ', log = ' + str(args.log) + ', nruns = ' + str(args.number) +
                ', re-insertion = ' + str(args.reinsertion) + '\n')
        f.write('# Best result: ' + str(mx_v) + ', duration = ' + str(mx_time) +  ' ')
        if is_bc:
            f.write('BC\n')
        else:
            f.write('BDC\n')
            
        avg = np.mean([min(res[1], res[2]) for res in out])
        avg_time = np.mean([res[3] for res in out])
        f.write('# avg: ' + str(avg) + ', avg duration = ' + str(avg_time) + '\n')
        f.write('# iteration No., bc value, bdc value, duration\n')
        for res in out:
            f.write(str(res[0]) + ',' + str(res[1]) + ',' + str(res[2]) + ',' + str(res[3]) + '\n')

    print(mx_v, avg, mx_time)
    
if __name__ == '__main__':
    sys.setrecursionlimit(int(1e9))
    main()
