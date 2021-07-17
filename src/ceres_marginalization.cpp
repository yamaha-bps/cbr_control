// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include "cbr_control/pge/ceres_marginalization.hpp"

#include <ceres/ceres.h>

#include <functional>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cbr
{

bool marginalize(
  const ceres::Problem * problem, const double * node,
  std::vector<MarginalizationInfo> & varinfo,
  Eigen::VectorXd & gamma, Eigen::MatrixXd & sqrt_inf)
{
  using std::vector;
  using Eigen::MatrixXd, Eigen::VectorXd;
  using MatrixXdRowM = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

  // Find residual blocks F_list and set of neighbords a_set
  vector<ceres::ResidualBlockId> F_list;
  std::unordered_set<const double *> a_set{node};

  problem->GetResidualBlocksForParameterBlock(node, &F_list);
  for (const auto & resblock : F_list) {
    vector<double *> resblock_neighbors;
    problem->GetParameterBlocksForResidualBlock(resblock, &resblock_neighbors);
    a_set.insert(resblock_neighbors.begin(), resblock_neighbors.end());
  }

  // TODO(pettni): add option to marginalize all factors in the neighbor clique,
  // as it is now only factors connected to the marginalized node are removed
  // do this by adding all factors between nodes in a_set

  a_set.erase(node);

  if (a_set.empty()) {
    return false;  // node isolated, nothing to marginalize
  }

  // two partitions of variables (b is to be marginalized out)
  vector<const double *> a_list(a_set.begin(), a_set.end());
  vector<const double *> b_list{node};

  /* map node address -> index in a_list */
  std::unordered_map<const double *, int> a_idx;
  std::unordered_map<const double *, int> b_idx;

  /* size of node blocks (local parameterization) */
  vector<int> a_len(a_list.size(), 0);
  vector<int> b_len(b_list.size(), 0);

  /* global length */
  vector<int> a_glen(a_list.size(), 0);

  for (size_t i = 0; i != a_list.size(); ++i) {
    a_idx[a_list[i]] = i;
    a_len[i] = problem->ParameterBlockLocalSize(a_list[i]);
    a_glen[i] = problem->ParameterBlockSize(a_list[i]);
  }

  for (size_t i = 0; i != b_list.size(); ++i) {
    b_idx[b_list[i]] = i;
    b_len[i] = problem->ParameterBlockLocalSize(b_list[i]);
  }

  // calculate starting indices of blocks in local parameterization
  vector<int> a_sta{0};
  std::partial_sum(a_len.begin(), a_len.end(), std::back_inserter(a_sta), std::plus<int>());
  vector<int> b_sta{0};
  std::partial_sum(b_len.begin(), b_len.end(), std::back_inserter(b_sta), std::plus<int>());

  // form partitioned information pair
  // [ I_aa   I_ab ]     [ eta_a ]
  // [ I_ba   I_bb ]     [ eta_b ]
  int a_d = std::accumulate(a_len.begin(), a_len.end(), 0, std::plus<int>());
  int b_d = std::accumulate(b_len.begin(), b_len.end(), 0, std::plus<int>());
  MatrixXd I_aa = MatrixXd::Zero(a_d, a_d);
  MatrixXd I_ab = MatrixXd::Zero(a_d, b_d);
  MatrixXd I_bb = MatrixXd::Zero(b_d, b_d);
  VectorXd eta_a = VectorXd::Zero(a_d);
  VectorXd eta_b = VectorXd::Zero(b_d);

  // fill in with contributions from each block
  for (const auto f_j : F_list) {
    const ceres::CostFunction * cost_fcn = problem->GetCostFunctionForResidualBlock(f_j);

    vector<double *> I_j;
    problem->GetParameterBlocksForResidualBlock(f_j, &I_j);

    // prepare cost and jacobians
    VectorXd h_j(cost_fcn->num_residuals());
    vector<MatrixXdRowM> dh_g(I_j.size());
    vector<MatrixXdRowM> dh_l(I_j.size());
    vector<double *> dh_g_ptrs(I_j.size());
    for (size_t i = 0; i != I_j.size(); ++i) {
      dh_l[i] = MatrixXdRowM(cost_fcn->num_residuals(), problem->ParameterBlockLocalSize(I_j[i]));
      dh_g[i] = MatrixXdRowM(cost_fcn->num_residuals(), problem->ParameterBlockSize(I_j[i]));
      dh_g_ptrs[i] = dh_g[i].data();
    }

    // evaluate cost and jacobians for this block
    cost_fcn->Evaluate(I_j.data(), h_j.data(), dh_g_ptrs.data());

    // convert jacobians to local parameterization
    for (size_t i = 0; i != I_j.size(); ++i) {
      auto prm = problem->GetParameterization(I_j[i]);
      if (prm != NULL) {
        prm->MultiplyByJacobian(I_j[i], cost_fcn->num_residuals(), dh_g[i].data(), dh_l[i].data());
      } else {
        dh_l[i] = dh_g[i];
      }
    }

    for (size_t i1 = 0; i1 != I_j.size(); ++i1) {
      double * k1 = I_j[i1];

      if (a_idx.find(k1) != a_idx.end()) {
        // i1 is an a variable
        int s1 = a_sta[a_idx[k1]];
        int l1 = a_len[a_idx[k1]];
        eta_a.segment(s1, l1) += h_j.transpose() * dh_l[i1];
      } else if (b_idx.find(k1) != b_idx.end()) {
        // i1 is a b variable
        int s1 = b_sta[b_idx[k1]];
        int l1 = b_len[b_idx[k1]];
        eta_b.segment(s1, l1) += h_j.transpose() * dh_l[i1];
      } else {
        throw std::runtime_error("Not a valid segment");
      }

      for (size_t i2 = i1; i2 != I_j.size(); ++i2) {
        double * k2 = I_j[i2];
        if (a_idx.find(k1) != a_idx.end() && a_idx.find(k2) != a_idx.end()) {
          // inside I_aa
          int s1 = a_sta[a_idx[k1]];
          int l1 = a_len[a_idx[k1]];
          int s2 = a_sta[a_idx[k2]];
          int l2 = a_len[a_idx[k2]];
          I_aa.block(s1, s2, l1, l2) += dh_l[i1].transpose() * dh_l[i2];
          if (s1 != s2) {
            // off diagonal: fill symmetric part
            I_aa.block(s2, s1, l2, l1) += dh_l[i2].transpose() * dh_l[i1];
          }
        } else if (a_idx.find(k1) != a_idx.end() && b_idx.find(k2) != b_idx.end()) {
          // inside I_ab
          int s1 = a_sta[a_idx[k1]];
          int l1 = a_len[a_idx[k1]];
          int s2 = b_sta[b_idx[k2]];
          int l2 = b_len[b_idx[k2]];
          I_ab.block(s1, s2, l1, l2) += dh_l[i1].transpose() * dh_l[i2];
        } else if (b_idx.find(k1) != b_idx.end() && a_idx.find(k2) != a_idx.end()) {
          // inside I_ba
          int s1 = b_sta[b_idx[k1]];
          int l1 = b_len[b_idx[k1]];
          int s2 = a_sta[a_idx[k2]];
          int l2 = a_len[a_idx[k2]];
          I_ab.block(s2, s1, l2, l1) += dh_l[i2].transpose() * dh_l[i1];
        } else if (b_idx.find(k1) != b_idx.end() && b_idx.find(k2) != b_idx.end()) {
          // inside I_bb
          int s1 = b_sta[b_idx[k1]];
          int l1 = b_len[b_idx[k1]];
          int s2 = b_sta[b_idx[k2]];
          int l2 = b_len[b_idx[k2]];
          I_bb.block(s1, s2, l1, l2) += dh_l[i1].transpose() * dh_l[i2];
          if (s1 != s2) {
            I_bb.block(s2, s1, l2, l1) += dh_l[i2].transpose() * dh_l[i1];
          }
        } else {
          throw std::runtime_error("Not a valid block");
        }
      }
    }
  }

  // marginalize information form via Schur complement
  VectorXd eta = eta_a - I_ab * I_bb.inverse() * eta_b;
  MatrixXd Lambda = I_aa - I_ab * I_bb.inverse() * I_ab.transpose();

  // separate out full-rank part
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(Lambda);
  int N = es.eigenvalues().size();
  int rank = N;
  while (rank > 0 && es.eigenvalues()(N - rank) < 1e-5) {
    --rank;
  }

  if (rank < 1) {
    return false;
  }

  const VectorXd Ddiag = es.eigenvalues().segment(N - rank, rank);
  const MatrixXd U = es.eigenvectors().block(0, N - rank, N, rank);

  sqrt_inf = Ddiag.cwiseSqrt().asDiagonal() * U.transpose();
  gamma = -U * Ddiag.cwiseInverse().asDiagonal() * U.transpose() * eta;

  varinfo.clear();
  std::transform(
    a_list.begin(), a_list.end(), std::back_inserter(varinfo),
    [&a_sta, &a_idx, &a_len, &a_glen](const double * node) {
      return MarginalizationInfo(
        node, a_sta[a_idx[node]], a_len[a_idx[node]], a_glen[a_idx[node]],
        Eigen::Map<const VectorXd>(node, a_glen[a_idx[node]])
      );
    });

  return true;
}

}   // namespace cbr
