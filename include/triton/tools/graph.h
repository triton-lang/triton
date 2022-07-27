#pragma once

#ifndef _TRITON_TOOLS_THREAD_GRAPH_H_
#define _TRITON_TOOLS_THREAD_GRAPH_H_

#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace triton {
namespace tools {

template <class node_t> class graph {
  typedef std::map<node_t, std::set<node_t>> edges_t;

public:
  typedef std::map<size_t, std::vector<node_t>> cmap_t;
  typedef std::map<node_t, size_t> nmap_t;

private:
  void connected_components_impl(node_t x, std::set<node_t> &nodes,
                                 nmap_t *nmap, cmap_t *cmap, int id) const {
    if (nmap)
      (*nmap)[x] = id;
    if (cmap)
      (*cmap)[id].push_back(x);
    if (nodes.find(x) != nodes.end()) {
      nodes.erase(x);
      for (const node_t &y : edges_.at(x))
        connected_components_impl(y, nodes, nmap, cmap, id);
    }
  }

public:
  void connected_components(cmap_t *cmap, nmap_t *nmap) const {
    if (cmap)
      cmap->clear();
    if (nmap)
      nmap->clear();
    std::set<node_t> nodes = nodes_;
    unsigned id = 0;
    while (!nodes.empty()) {
      connected_components_impl(*nodes.begin(), nodes, nmap, cmap, id++);
    }
  }

  void add_edge(node_t x, node_t y) {
    nodes_.insert(x);
    nodes_.insert(y);
    edges_[x].insert(y);
    edges_[y].insert(x);
  }

  void clear() {
    nodes_.clear();
    edges_.clear();
  }

private:
  std::set<node_t> nodes_;
  edges_t edges_;
};

} // namespace tools
} // namespace triton

#endif
