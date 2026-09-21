/*-------------------------------------------------------------------------------
  Copyright (c) 2024 GRF Contributors.

  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <iterator>
#include "sampling/RandomSampler.h"

#include "tree/Tree.h"
#include "commons/utility.h"

namespace grf {

Tree::Tree(size_t root_node,
           const std::vector<std::vector<size_t>>& child_nodes,
           const std::vector<std::vector<size_t>>& leaf_samples,
           const std::vector<size_t>& split_vars,
           const std::vector<double>& split_values,
           const std::vector<size_t>& drawn_samples,
           const std::vector<bool>& send_missing_left,
           const PredictionValues& prediction_values) :
    root_node(root_node),
    child_nodes(child_nodes),
    leaf_samples(leaf_samples),
    split_vars(split_vars),
    split_values(split_values),
    drawn_samples(drawn_samples),
    send_missing_left(send_missing_left),
    prediction_values(prediction_values) {}

size_t Tree::get_root_node() const {
  return root_node;
}

const std::vector<std::vector<size_t>>& Tree::get_child_nodes() const {
  return child_nodes;
}

const std::vector<std::vector<size_t>>& Tree::get_leaf_samples() const {
  return leaf_samples;
}

const std::vector<size_t>& Tree::get_split_vars() const  {
  return split_vars;
}

const std::vector<double>& Tree::get_split_values() const  {
  return split_values;
}

const std::vector<size_t>& Tree::get_drawn_samples() const  {
  return drawn_samples;
}

const std::vector<bool>& Tree::get_send_missing_left() const  {
  return send_missing_left;
}

const PredictionValues& Tree::get_prediction_values() const  {
  return prediction_values;
}

std::vector<size_t> Tree::find_leaf_nodes(const Data& data,
                                          const std::vector<size_t>& samples) const  {
  std::vector<size_t> prediction_leaf_nodes;
  prediction_leaf_nodes.resize(data.get_num_rows());

  for (size_t sample : samples) {
    size_t node = find_leaf_node(data, sample);
    prediction_leaf_nodes[sample] = node;
  }
  return prediction_leaf_nodes;
}

std::vector<size_t> Tree::find_leaf_nodes(const Data& data,
                                          const std::vector<bool>& valid_samples) const  {
  size_t num_samples = data.get_num_rows();

  std::vector<size_t> prediction_leaf_nodes;
  prediction_leaf_nodes.resize(num_samples);

  for (size_t sample = 0; sample < num_samples; sample++) {
    if (!valid_samples[sample]) {
      continue;
    }

    size_t node = find_leaf_node(data, sample);
    prediction_leaf_nodes[sample] = node;
  }
  return prediction_leaf_nodes;
}

void Tree::set_leaf_samples(const std::vector<std::vector<size_t>>& leaf_samples) {
  this->leaf_samples = leaf_samples;
}

void Tree::set_prediction_values(const PredictionValues& prediction_values) {
  this->prediction_values = prediction_values;
}


size_t Tree::find_leaf_node(const Data& data,
                            size_t sample) const  {
  return find_leaf_node(data, sample, root_node);
}

size_t Tree::find_leaf_node(const Data& data,
                            size_t sample,
                            size_t start_node) const  {
  size_t node = start_node;
  while (true) {
    // Break if terminal node
    if (is_leaf(node)) {
      break;
    }

    // Move to child
    size_t split_var = get_split_vars()[node];
    double split_val = get_split_values()[node];
    double value = data.get(sample, split_var);
    bool send_na_left = get_send_missing_left()[node];
    if (
        (value <= split_val) || // ordinary split
        (send_na_left && std::isnan(value)) || // are we sending NaN left
        (std::isnan(split_val) && std::isnan(value)) // are we splitting on NaN
      ) {
      // Move to left child
      node = child_nodes[0][node];
    } else {
      // Move to right child
      node = child_nodes[1][node];
    }
  }
  return node;
};

void Tree::honesty_prune_leaves(const Data& data, size_t min_leaf_samples) {
  size_t num_nodes = leaf_samples.size();
  for (size_t n = num_nodes; n > root_node; n--) {
    size_t node = n - 1;
    if (is_leaf(node)) {
      continue;
    }

    prune_node(child_nodes[0][node], data, min_leaf_samples);
    prune_node(child_nodes[1][node], data, min_leaf_samples);
  }
  prune_node(root_node, data, min_leaf_samples);
}

void Tree::prune_node(size_t& node, const Data& data, size_t min_leaf_samples) {
  if (is_leaf(node)) {
    return;
  }

  size_t left_child = child_nodes[0][node];
  size_t right_child = child_nodes[1][node];

  bool prune_left = is_small_leaf(left_child, min_leaf_samples);
  bool prune_right = is_small_leaf(right_child, min_leaf_samples);

  // If both children hold enough samples, there is nothing to prune.
  if (!prune_left && !prune_right) {
    return;
  }

  // Empty out this node.
  child_nodes[0][node] = 0;
  child_nodes[1][node] = 0;

  if (prune_left && prune_right) {
    // Both children are too small: collapse them into this node. The merged node may still
    // be too small, in which case it is pruned in turn when its own parent is visited.
    reroute_samples(data, left_child, node);
    reroute_samples(data, right_child, node);
  } else if (prune_left) {
    // Promote the right subtree, sending the left child's samples down it.
    reroute_samples(data, left_child, right_child);
    node = right_child;
  } else {
    reroute_samples(data, right_child, left_child);
    node = left_child;
  }
}

void Tree::reroute_samples(const Data& data,
                           size_t from_node,
                           size_t to_node) {
  std::vector<size_t> samples;
  samples.swap(leaf_samples[from_node]);

  for (size_t sample : samples) {
    // The destination may be an entire subtree, so the sample is sent down its splits.
    size_t leaf_node = find_leaf_node(data, sample, to_node);
    leaf_samples[leaf_node].push_back(sample);
  }
}

bool Tree::is_leaf(size_t node) const  {
  return child_nodes[0][node] == 0 && child_nodes[1][node] == 0;
}

bool Tree::is_small_leaf(size_t node, size_t min_leaf_samples) const  {
  return is_leaf(node) && leaf_samples[node].size() < min_leaf_samples;
}

} // namespace grf
