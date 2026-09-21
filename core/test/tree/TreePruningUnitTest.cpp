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

#include "catch.hpp"
#include "commons/Data.h"
#include "tree/Tree.h"

using namespace grf;

namespace {

/**
 * A single covariate for samples 0, ..., 11: the samples in the first half take
 * the value 1.0, the ones in the second half 9.0.
 */
std::vector<double> covariate = {1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9};

/**
 * The leaf-size floor the pruning test cases are written against.
 */
const size_t MIN_LEAF_SAMPLES = 5;

} // namespace

TEST_CASE("pruning removes leaves that are too small", "[tree, unit]") {
  /*
   * This test case starts with the following tree structure, where the leaves
   * hold too few samples except for nodes 6 and 7:
   *
   *             0
   *           /   \
   *          1     2
   *        /   \
   *       3     4
   *      / \   / \
   *     5   6 7   8
   *              / \
   *             9  10
   *
   * Pruning should produce this tree (note the root node ID has changed):
   *
   *          1
   *         / \
   *        6   7
   */
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes =
      {{1, 3, 0, 5, 7, 0, 0, 0, 9, 0, 0}, {2, 4, 0, 6, 8, 0, 0, 0, 10, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {
      {}, {}, {}, {}, {}, {}, {0, 1, 2, 3, 4}, {6, 7, 8, 9, 10}, {}, {}, {}};
  Tree tree(0, child_nodes, leaf_nodes, std::vector<size_t>(11, 0),
            std::vector<double>(11, 0), {0}, std::vector<bool>(11, true), PredictionValues());

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);

  std::vector<std::vector<size_t>> expected_child_nodes =
      {{0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  const std::vector<std::vector<size_t>>& actual_child_nodes = tree.get_child_nodes();

  REQUIRE(tree.get_root_node() == 1);
  for (size_t node = 0; node < leaf_nodes.size(); node++) {
    REQUIRE(actual_child_nodes[0][node] == expected_child_nodes[0][node]);
    REQUIRE(actual_child_nodes[1][node] == expected_child_nodes[1][node]);
  }
}

TEST_CASE("a leaf-size floor of one prunes only the empty leaves", "[tree, unit]") {
  /*
   * The same tree as above, but with the two populated leaves holding a single sample
   * each. A floor of one is the default, and leaves them in place.
   */
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes =
      {{1, 3, 0, 5, 7, 0, 0, 0, 9, 0, 0}, {2, 4, 0, 6, 8, 0, 0, 0, 10, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {
      {}, {}, {}, {}, {}, {}, {0}, {6}, {}, {}, {}};
  Tree tree(0, child_nodes, leaf_nodes, std::vector<size_t>(11, 0),
            std::vector<double>(11, 0), {0}, std::vector<bool>(11, true), PredictionValues());

  tree.honesty_prune_leaves(data, 1);

  std::vector<std::vector<size_t>> expected_child_nodes =
      {{0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  const std::vector<std::vector<size_t>>& actual_child_nodes = tree.get_child_nodes();

  REQUIRE(tree.get_root_node() == 1);
  for (size_t node = 0; node < leaf_nodes.size(); node++) {
    REQUIRE(actual_child_nodes[0][node] == expected_child_nodes[0][node]);
    REQUIRE(actual_child_nodes[1][node] == expected_child_nodes[1][node]);
  }
  REQUIRE(tree.get_leaf_samples()[6] == std::vector<size_t> {0});
  REQUIRE(tree.get_leaf_samples()[7] == std::vector<size_t> {6});
}

TEST_CASE("pruning reroutes samples into the sibling leaf", "[tree, unit]") {
  /*
   *      0                          Node 1 is too small, so it is pruned and its
   *     / \       ->      2         samples are merged into its sibling.
   *    1   2
   */
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes = {{1, 0, 0}, {2, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {{}, {0, 1, 2}, {6, 7, 8, 9, 10}};
  Tree tree(0, child_nodes, leaf_nodes, {0, 0, 0}, {0, 0, 0}, {0}, {true, true, true},
            PredictionValues());

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);

  REQUIRE(tree.get_root_node() == 2);
  REQUIRE(tree.get_leaf_samples()[1].empty());
  REQUIRE(tree.get_leaf_samples()[2] == std::vector<size_t> {6, 7, 8, 9, 10, 0, 1, 2});
}

TEST_CASE("pruning reroutes samples through the sibling subtree", "[tree, unit]") {
  /*
   *      0                       2           Node 1 is too small, so it is pruned and
   *     / \                     / \          its samples are sent down the splits of
   *    1   2         ->        3   4         the subtree that takes its place.
   *       / \
   *      3   4
   */
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes = {{1, 0, 3, 0, 0}, {2, 0, 4, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {
      {}, {5, 6}, {}, {0, 1, 2, 3, 4}, {7, 8, 9, 10, 11}};
  // Node 2 splits on the only covariate at 5.0, sending sample 5 left and sample 6 right.
  Tree tree(0, child_nodes, leaf_nodes, {0, 0, 0, 0, 0}, {0, 0, 5, 0, 0}, {0},
            std::vector<bool>(5, true), PredictionValues());

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);

  REQUIRE(tree.get_root_node() == 2);
  REQUIRE(tree.get_leaf_samples()[1].empty());
  REQUIRE(tree.get_leaf_samples()[3] == std::vector<size_t> {0, 1, 2, 3, 4, 5});
  REQUIRE(tree.get_leaf_samples()[4] == std::vector<size_t> {7, 8, 9, 10, 11, 6});
}

TEST_CASE("pruning collapses two small siblings into their parent", "[tree, unit]") {
  /*
   *        0                                  Nodes 3 and 4 are both too small, so they
   *       / \                                 collapse into node 1. The merged node is
   *      1   2          ->          2         still too small, so it is in turn merged
   *     / \                                   into node 2.
   *    3   4
   */
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes = {{1, 3, 0, 0, 0}, {2, 4, 0, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {
      {}, {}, {6, 7, 8, 9, 10}, {0, 1}, {2, 3}};
  Tree tree(0, child_nodes, leaf_nodes, std::vector<size_t>(5, 0), std::vector<double>(5, 0),
            {0}, std::vector<bool>(5, true), PredictionValues());

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);

  REQUIRE(tree.get_root_node() == 2);
  REQUIRE(tree.get_leaf_samples()[1].empty());
  REQUIRE(tree.get_leaf_samples()[3].empty());
  REQUIRE(tree.get_leaf_samples()[4].empty());
  REQUIRE(tree.get_leaf_samples()[2] == std::vector<size_t> {6, 7, 8, 9, 10, 0, 1, 2, 3});
}

TEST_CASE("pruning keeps a root that is too small", "[tree, unit]") {
  /*
   * The whole tree holds fewer than the leaf-size floor, so the two leaves
   * collapse into the root, which is left in place.
   */
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes = {{1, 0, 0}, {2, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {{}, {0, 1}, {6}};
  Tree tree(0, child_nodes, leaf_nodes, {0, 0, 0}, {0, 0, 0}, {0}, {true, true, true},
            PredictionValues());

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);

  REQUIRE(tree.get_root_node() == 0);
  REQUIRE(tree.get_child_nodes()[0][0] == 0);
  REQUIRE(tree.get_child_nodes()[1][0] == 0);
  REQUIRE(tree.get_leaf_samples()[0] == std::vector<size_t> {0, 1, 6});
}

TEST_CASE("pruning is idempotent", "[tree, unit]") {
  Data data(covariate, 12, 1);

  std::vector<std::vector<size_t>> child_nodes =
      {{1, 3, 0, 5, 7, 0, 0, 0, 9, 0, 0}, {2, 4, 0, 6, 8, 0, 0, 0, 10, 0, 0}};
  std::vector<std::vector<size_t>> leaf_nodes = {
      {}, {}, {}, {}, {}, {}, {0, 1, 2, 3, 4}, {6, 7, 8, 9, 10}, {}, {}, {}};
  Tree tree(0, child_nodes, leaf_nodes, std::vector<size_t>(11, 0),
            std::vector<double>(11, 0), {0}, std::vector<bool>(11, true), PredictionValues());

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);
  std::vector<std::vector<size_t>> expected_child_nodes = tree.get_child_nodes();
  std::vector<std::vector<size_t>> expected_leaf_samples = tree.get_leaf_samples();
  size_t expected_root_node = tree.get_root_node();

  tree.honesty_prune_leaves(data, MIN_LEAF_SAMPLES);

  REQUIRE(tree.get_root_node() == expected_root_node);
  REQUIRE(tree.get_child_nodes() == expected_child_nodes);
  REQUIRE(tree.get_leaf_samples() == expected_leaf_samples);
}
