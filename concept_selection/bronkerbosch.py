MIN_SIZE = 13


# coding: utf-8

class Reporter(object):
    def __init__(self):
        self.cnt = 0
        self.cliques = []
 
    def inc_count(self):
        self.cnt += 1
 
    def record(self, clique):
        self.cliques.append(clique)
 
    def print_report(self):
        print ('%d recursive calls' % self.cnt)
        for i, clique in enumerate(self.cliques):
            print ('%d: %s' % (i, clique))
        print()

    def get_cliques(self):
        return self.cliques

def bronker_bosch2(clique, candidates, excluded, reporter, NEIGHBORS):
    '''Bron–Kerbosch algorithm with pivot'''
    reporter.inc_count()
    if not candidates and not excluded:
        if len(clique) >= MIN_SIZE:
            reporter.record(clique)
        return
 
    pivot = pick_random(candidates) or pick_random(excluded)
    for v in list(candidates.difference(NEIGHBORS[pivot])):
        new_candidates = candidates.intersection(NEIGHBORS[v])
        new_excluded = excluded.intersection(NEIGHBORS[v])
        bronker_bosch2(clique + [v], new_candidates, new_excluded, reporter, NEIGHBORS)
        candidates.remove(v)
        excluded.add(v)

def pick_random(s):
    if s:
        elem = s.pop()
        s.add(elem)
        return elem
    

# NEIGHBORS = [
#     [], # I want to start index from 1 instead of 0
#     [2,3,7],
#     [1,3,4,6,7],
#     [1,2,8],
#     [2,6,5,8],
#     [4,6],
#     [2,4,5],
#     [1,2],
#     [3,4],
#     [8]
# ]
# NODES = set(range(1, len(NEIGHBORS)))

# report = Reporter()
# bronker_bosch2([], set(NODES), set(), report)
# report.print_report()