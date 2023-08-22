from __future__ import annotations
import sys, os
import copy
from time import time

external_utility = dict()
file = None
counter, candidates = 0, 0

# decorator for benchmarking
def benchmark(func):
  def count_time(*args, **kwargs):
    start = time()
    func(*args, **kwargs)
    stop = time()
    global file
    file.write("Time taken: %.3fs" %(stop - start))
    return
  return count_time

class HyperEdge:
  def __init__(self, id=None, label=None, vertices: dict=None) -> None:
    self.id = id
    self.label = label
    self.vertices = dict() if vertices == None else vertices
  
  # adds a vertex to vertices
  def add_vertex(self, v_id, v_utility='-') -> None:
    if self.contains(id):
      raise Exception("Vertex already exists")
    self.vertices[v_id] = v_utility
  
  # checks if the edge contains a vertex given an id
  def contains(self, v_id : int) -> bool:
    if v_id not in self.vertices:
      return False
    
    return True
  
  def __eq__(self, other) -> bool:
    if isinstance(other, HyperEdge):
      return self.id == other.id and self.label == other.label and self.vertices == other.vertices
    
    return False
  
  def __str__(self) -> str:
    return "{}, {}, {}".format(self.id, self.label, self.vertices)

class HyperGraph:
  def __init__(self) -> None:
    self.vertices = dict()
    self.edges = dict()
    self.vertex_count = 0
    self.e_labels = dict()
    self.v_labels = dict()
    self.v_component = dict()
  
  # list of ids of vertices
  def vertex_ids(self) -> list[int]:
    return list(self.vertices.keys())

  # list of ids of edges
  def edge_ids(self) -> list[int]:
    return list(self.edges.keys())

  # list of edges
  def get_edges(self) -> list:
    return list(self.edges.values())
  
  # finds the component of v using dsu and path compression
  def find_component(self, v):
    if self.v_component[v] == v:
      return v
    
    comp = self.find_component(self.v_component[v])
    self.v_component[v] = comp
    return comp
  
  # adds a vertex to the hypergraph
  def add_vertex(self, v_id, v_label) -> None:
    if(v_id in self.vertices and self.vertices[v_id] == v_label):
      return 
    
    if(v_id in self.vertices):
      raise Exception('Vertex with same id already exists.')
     
    self.vertices[v_id] = v_label
    self.v_component[v_id] = v_id
    self.vertex_count += 1
    
    if v_label in self.v_labels:
      self.v_labels[v_label] += 1
    else:
      self.v_labels[v_label] = 1
  
  # adds a hyperedge to the hypergraph
  def add_edge(self, edge: HyperEdge) -> None:
    if(edge.id in self.edges and self.edges[edge.id] == edge):
      return 
    
    if(edge.id in self.edges):
      raise Exception('Edge with same id already exists.')
    
    components = []
    for v in edge.vertices:
      if v not in self.vertices:
        raise Exception('Vertex doesn\'t exist in hypergraph.')
      
      components.append(self.find_component(v))
      
    for c in components:
      self.v_component[c] = components[0]
    
    self.edges[edge.id] = edge
    
    if edge.label in self.e_labels:
      self.e_labels[edge.label] += 1
    else:
      self.e_labels[edge.label] = 1
  
  # returns the utility of the hypergraph
  def utility(self) -> float:
    utility = 0
    for e in self.get_edges():
      edge_utility = 0
      for v in e.vertices:
        edge_utility += e.vertices[v] * external_utility[self.vertices[v]]
    
      utility += edge_utility

    return utility
  
  # returns the utility of the subhypergraph that is isomorphic to h
  def subhypergraph_utility(self, h : HyperGraph, phi_v: list[int], phi_e: list[int]) -> float:
    utility = 0
    for e in self.get_edges():
      edge_utility = 0
      if e.id in phi_e:    
        e_h = h.edges[phi_e[e.id]]
        for v in e.vertices:
          if v not in phi_v:
            continue
          elif e_h.contains(phi_v[v]):
            edge_utility += e.vertices[v] * external_utility[self.vertices[v]]
            
      utility += edge_utility
      
    return utility
  
  # returns the cwu pruning value
  def cwu(self, vertex) -> float:
    utility = 0
    component = self.find_component(vertex)
    
    for e in self.get_edges():
      edge_utility = 0
      for v in e.vertices:
        if component != self.find_component(v):
          continue
        vertex_utility = e.vertices[v] * external_utility[self.vertices[v]]
        utility += vertex_utility
    
    return utility     
  
  # checks if h is an isomorphic graph candidate based on vertex and edge labels
  def can_isomorph(self, H) -> bool:
      
    if len(self.v_labels) > len(H.v_labels):
      return False
    if len(self.e_labels) > len(H.e_labels):
      return False
    
    for v_label in self.v_labels:
      if v_label not in H.v_labels:
        return False
      if H.v_labels[v_label] < self.v_labels[v_label]:
        return False
      
    for e_label in self.e_labels:
      if e_label not in H.e_labels:
        return False
      if H.e_labels[e_label] < self.e_labels[e_label]:
        return False
      
    return True
  
  def v_candidates(self, H, h_id):
    adj = dict()
    for at_h in range(len(h_id)):
      for v in H.vertices:
        if self.vertices[h_id[at_h]] == H.vertices[v]:
          if h_id[at_h] in adj:
            adj[h_id[at_h]].append(v)
          else:
            adj[h_id[at_h]] = [v]

      if h_id[at_h] not in adj:
        return []
    return adj
  
  def e_candidates(self, H, h_id, phi):
    adj = dict()
    for at_h in range(len(h_id)):
      for e in H.edges:
        if self.edges[h_id[at_h]].label == H.edges[e].label:
          flag = True
          for v in self.edges[h_id[at_h]].vertices:
            if phi[v] not in H.edges[e].vertices:
              flag = False
              break
          
          if flag:
            if h_id[at_h] in adj:
              adj[h_id[at_h]].append(e)
            else:
              adj[h_id[at_h]] = [e]
      
      if h_id[at_h] not in adj:
        return []        
    return adj
  
  # calculates the mapping functions for isomorphism
  def isomorphic_mapping(self, H) -> tuple(int, int):
    
    # phi = self -> H, phi_inv = H -> self
    phi_vs, phi_inv_vs = [], []
    phi_es, phi_inv_es = [], []
    
    if self.can_isomorph(H):
      vh_id = self.vertex_ids()
      adj_v = self.v_candidates(H, vh_id)
      
      if len(adj_v) < len(vh_id):
        return [], [], [], []
      
      _phi_vs, _phi_inv_vs = self.map_v(H, 0, vh_id, dict(), dict(), adj_v)
      for at_v in range(0, len(_phi_vs)):
        eh_id = self.edge_ids()
        adj_e = self.e_candidates(H, eh_id, _phi_vs[at_v])

        if len(adj_e) < len(eh_id):
          continue
        
        _phi_es, _phi_inv_es = self.map_e(H, 0, self.edge_ids(), dict(), dict(), adj_e)
        for at_e in range(0, len(_phi_es)):
          phi_vs.append(_phi_vs[at_v])
          phi_inv_vs.append(_phi_inv_vs[at_v])
          phi_es.append(_phi_es[at_e])
          phi_inv_es.append(_phi_inv_es[at_e])
    
    return phi_vs, phi_inv_vs, phi_es, phi_inv_es
  
  # vertex mapping
  def map_v(self, H, at_h, h_id, match, inv_match, adj) -> tuple:
    if at_h == len(self.vertices):
      return [match], [inv_match]
    
    phi_vs, phi_inv_vs = [], []
    v_h = h_id[at_h]
    
    for v_H in adj[v_h]:
      if v_H in inv_match:
        continue
      
      _match = copy.deepcopy(match)
      _inv_match = copy.deepcopy(inv_match)
      _match[v_h] = v_H
      _inv_match[v_H] = v_h
      
      _phis, phi_invs = self.map_v(H, at_h + 1, h_id, _match, _inv_match, adj)
      for at in range(0, len(_phis)):
        phi_vs.append(_phis[at])
        phi_inv_vs.append(phi_invs[at])

    return phi_vs, phi_inv_vs
  
  # edge mapping
  def map_e(self, H, at_h, e_id, match, inv_match, adj) -> tuple:
    if at_h == len(self.edges):
      return [match], [inv_match]
    
    phi_es, phi_inv_es = [], []
    e_h = e_id[at_h]
    
    for e_H in adj[e_h]:
      if e_H in inv_match:
        continue
            
      _match = copy.deepcopy(match)
      _inv_match = copy.deepcopy(inv_match)
      _match[e_h] = e_H
      _inv_match[e_H] = e_h
      
      _phis, phi_invs = self.map_e(H, at_h + 1, e_id, _match, _inv_match, adj)
      for at in range(0, len(_phis)):
        phi_es.append(_phis[at])
        phi_inv_es.append(phi_invs[at])

    return phi_es, phi_inv_es
  
  def __str__(self) -> str:
    s = 'vertex_id, vertex_label\n'
    for v in self.vertices:
      s += "{}, {}\n".format(v, self.vertices[v])
    s += '\n'
    s += 'edge_id, edge_label, edge_utility, edge_vertices\n'
    for e in self.get_edges():
      s += e.__str__() + '\n'
    return s

class ExtensionTuple:
  def __init__(self, op=None, vertex=None, vertex_label=None, edge_label=None) -> None:
    
    self.op = op
    self.vertex = vertex
    self.vertex_label = vertex_label
    self.edge_label = edge_label
    
  def __str__(self) -> str:
    return "<{}, {}, {}, {}>".format(self.op, self.vertex, self.vertex_label, self.edge_label)
  
  def __lt__(self, other) -> bool:
    if self.op != other.op:
      return self.op == 'e'
      
    if self.op == 'a' and other.op == 'a':
      return (self.vertex, self.edge_label, self.vertex_label) < (other.vertex, other.edge_label, other.vertex_label)
    
    if self.op == 'e' and other.op == 'e':
      return (self.vertex, self.vertex_label) < (other.vertex, other.vertex_label)
  
  def __eq__(self, other) -> bool:
    
    if isinstance(other, ExtensionTuple):
      return self.op == other.op and self.vertex == other.vertex and self.vertex_label == other.vertex_label and self.edge_label == other.edge_label
    else:
      return False
  
  def __hash__(self) -> int:
    return hash((self.op, self.vertex, self.vertex_label, self.edge_label))

class Code:
  def __init__(self) -> None:
    self.tuples = []
    self.last_v = None
    self.last_e = None
    self.hypergraph = None
    self.added_vertex = []
  
  # adds a tuple to the code
  def add_tuple(self, t : ExtensionTuple) -> None:
    global vertex_added
    vertex_added = False
    if self.last_v == None or t.vertex > self.last_v:
      if self.hypergraph == None:
        self.hypergraph = HyperGraph()
      
      self.added_vertex.append(True)
      self.last_v = t.vertex
      self.hypergraph.add_vertex(t.vertex, t.vertex_label)
      vertex_added = True
    
    if t.op == 'a':
      if self.last_e == None:
        self.last_e = 0
      else:
        self.last_e += 1
      
      self.hypergraph.add_edge(HyperEdge(self.last_e, t.edge_label))
    
    self.added_vertex.append(vertex_added)
    self.tuples.append(t)
    self.hypergraph.edges[self.last_e].add_vertex(t.vertex)
  
  # returns true if the code will still be a canonical 
  # candidate after adding the tuple, false otherwise
  def canonical_candidate(self, t: ExtensionTuple) -> bool:
    return True
  
  # checks whether last edge contains vertex
  def last_edge_contains(self, vertex) -> bool:
    if self.last_e == None:
      return False
    
    return self.hypergraph.edges[self.last_e].contains(vertex)

  def __str__(self) -> str:
    s = ''
    for t in self.tuples:
      s += t.__str__() + '\n'
    return s

def min(ext1, ext2) -> ExtensionTuple:
  if ext1 < ext2:
    return ext1
  return ext2

# from a list of extension tuples, returns the minimum
def get_min_ext(extensions : list[ExtensionTuple]) -> ExtensionTuple:
  ret = None
  for ext in extensions:
    if ret == None:
      ret = ext
    else:
      ret = min(ret, ext)

  return ret

# returns the possible extension tuples of code with utility and pruning value
def find_extensions(code : Code, hypergraphs, canonical=False) -> dict[ExtensionTuple: (float, float)] :
  extensions = dict()
  h = code.hypergraph
  at = 0
  for H in hypergraphs:
    at += 1
    _extensions = dict()
    if h == None:
      for e in H.get_edges():
        for v in e.vertices:
          ext = ExtensionTuple('a', 0, H.vertices[v], e.label)
          utility, pruning_value = 0, 0
          if canonical == False:
            utility = e.vertices[v] * external_utility[H.vertices[v]] 
            pruning_value = H.cwu(v)
            
          if ext in _extensions:
            utility = max(_extensions[ext][0], utility)
            pruning_value = max(_extensions[ext][1], pruning_value)

          _extensions[ext] = (utility, pruning_value)
            
    else:
      phi_vs, phi_inv_vs, phi_es, phi_inv_es = h.isomorphic_mapping(H)
      for i in range (0, len(phi_vs)):
        phi_inv_v = phi_inv_vs[i]
        phi_e, phi_inv_e = phi_es[i], phi_inv_es[i]
        
        for e in H.edges:
          e_label = H.edges[e].label
          for v in H.edges[e].vertices:
            v_label = H.vertices[v]
            if e not in phi_inv_e and v in phi_inv_v:
              ext = ExtensionTuple('a', phi_inv_v[v], v_label, e_label)
              if code.canonical_candidate(ext) == False:
                continue
              
              utility, pruning_value = 0, 0

              if canonical == False:
                new_code = copy.deepcopy(code)
                new_code.add_tuple(ext)
                new_phi_inv_e = copy.deepcopy(phi_inv_e)
                new_phi_inv_e[e] = new_code.last_e
                utility = H.subhypergraph_utility(new_code.hypergraph, phi_inv_v, new_phi_inv_e)
                
                pruning_value = H.cwu(v)

              if ext in _extensions:
                utility = max(_extensions[ext][0], utility)
                pruning_value = max(_extensions[ext][1], pruning_value)
              
              _extensions[ext] = (utility, pruning_value)
              
        e = H.edges[phi_e[code.last_e]]
        e_label = e.label
        for v in e.vertices:
          v_label = H.vertices[v]
          if v not in phi_inv_v:
            ext = ExtensionTuple('e', code.last_v + 1, v_label, e_label)
            if code.canonical_candidate(ext) == False:
              continue
            
            utility, pruning_value = 0, 0
            
            if canonical == False:
              new_code = copy.deepcopy(code)
              new_code.add_tuple(ext)
              new_phi_inv_v = copy.deepcopy(phi_inv_v)
              new_phi_inv_v[v] = new_code.last_v
              utility = H.subhypergraph_utility(new_code.hypergraph, new_phi_inv_v, phi_inv_e)
              
              pruning_value = H.cwu(v)
                  
            if ext in _extensions:
              utility = max(_extensions[ext][0], utility)
              pruning_value = max(_extensions[ext][1], pruning_value)
            
            _extensions[ext] = (utility, pruning_value)
          
          elif not code.last_edge_contains(phi_inv_v[v]):
            ext = ExtensionTuple('e', phi_inv_v[v], v_label, e_label)
            if code.canonical_candidate(ext) == False:
              continue
            
            utility, pruning_value = 0, 0
            
            if canonical == False:
              new_code = copy.deepcopy(code)
              new_code.add_tuple(ext)
              utility = H.subhypergraph_utility(new_code.hypergraph, phi_inv_v, phi_inv_e)
              
              pruning_value = H.cwu(v)
                  
            if ext in _extensions:
              utility = max(_extensions[ext][0], utility)
              pruning_value = max(_extensions[ext][1], pruning_value)
            
            _extensions[ext] = (utility, pruning_value)
      
    for ext in _extensions:
      _utility, _pruning_value = _extensions[ext]
      
      if ext in extensions:
        utility, pruning_value = extensions[ext]
        extensions[ext] = (utility + _utility, pruning_value + _pruning_value)
      else:
        extensions[ext] = (_utility, _pruning_value)
    
  return extensions

# checks if code is canonical
def is_canonical(code : Code) -> bool:
  h = code.hypergraph
  Ct = Code()
  
  for i in range(0, len(code.tuples)):
    exts = find_extensions(Ct, [h], True)
    min_ext = get_min_ext(exts)
    if code.tuples[i] != min_ext:
      return False
    
    Ct.add_tuple(code.tuples[i])
  
  return True

# Utility based hypergrah miner
def UHGMINER(code : Code, hypergraphs : list, min_util : float, lev = 0) -> None:
  global counter, candidates
  global file
  
  extensions = find_extensions(code, hypergraphs)
  at = 0
  for ext in extensions:
    at += 1
    
    candidates += 1
    utility, pruning_value = extensions[ext]

    next_code = copy.deepcopy(code)
    next_code.add_tuple(ext)
    if is_canonical(next_code) == False:
      continue
    
    if utility >= min_util:
      counter += 1
      file.write("High utility subhypergraph #{}\n\n".format(counter))
      file.write("Utility: %.3f\n\n" %(utility))
      file.write(next_code.hypergraph.__str__() + '\n')
      file.write("----------------------------------------------------------------\n\n")
      
    if pruning_value < min_util:
      continue
    
    if lev == 0: 
      print("{}: {} out of {}.".format(lev, at, len(extensions)))
    UHGMINER(next_code, hypergraphs, min_util, lev + 1)

# parse input from file
def parse_input(filename) -> list:
  folder = "datasets"
  file = open(folder + "/" + filename).readlines();
  
  hypergraphs = []
  hypergraph = HyperGraph()
  for line in file:
    if line == 'end\n':
      hypergraphs.append(hypergraph)
      hypergraph = HyperGraph()
      
    elif line[0] == 'u':
      _, label, utility = line.split()
      external_utility[int(label)] = float(utility)
      
    elif line[0] == 'v':
      _, id, label = line.split(' ')
      hypergraph.add_vertex(int(id), int(label))
            
    elif line[0] == 'e':
      tokens = line.split(' ')
      id = int(tokens[1])
      label = int(tokens[2])
      vertices = dict()
      for v in tokens[3 :]:
        if v == '\n':
          continue
        v_id, v_utility = v.split(',')
        v_id = int(v_id)
        v_utility = float(v_utility)
        vertices[v_id] = v_utility
      hypergraph.add_edge(HyperEdge(id, label, vertices))
  
  return hypergraphs

@benchmark
def main(filename, delta):

  hypergraphs = parse_input(filename)
  utility = 0.0
  for hypergraph in hypergraphs:
    utility += hypergraph.utility()

  global file
  global counter, candidates
  counter, candidates = 0, 0
  if file is not None:
    file.close()
  folder = "cwu_results"
  if not os.path.exists(folder):
    os.mkdir(folder)
  filename = filename.removesuffix('.txt')
  file = open(folder + "/" + '{}({:.4f}).txt'.format(filename, delta), 'w')
  
  UHGMINER(Code(), hypergraphs, utility * delta)
  
  file.write("Total utility: %.3f\n" %(utility))
  file.write("Minimum utility: %.3f\n" %(utility * delta))
  file.write("Number of high-utility subhypergraphs: {}\n".format(counter))
  file.write("Number of Candidates: {}\n".format(candidates))


filenames = {
  'ecommerce' : 'ecommerce.txt', 
  'foodmart' : 'foodmart.txt', 
  'fruithut' : 'fruithut.txt', 
  'liquor' : 'liquor.txt', 
  'bioinformatics' : 'Bioinformatics.txt', 
  'computer_network' : 'Computer_network.txt', 
  'computer_security' : 'Computer_security.txt', 
  'data_mining' : 'Data_mining.txt', 
  'distributed_computing' : 'Distributed_computing.txt', 
  'machine_learning' : 'Machine_learning.txt'
}

dataset = sys.argv[1]
threshold = sys.argv[2]

if dataset not in filenames:
  print("Invalid dataset name.")
  exit()

try:
  delta = float(threshold)
except:
  print("Invalid threshold value.")
  exit()
  
if delta < 0 or delta > 1:
  print("Invalid threshold value.")
  exit()

main(filenames[dataset], delta)