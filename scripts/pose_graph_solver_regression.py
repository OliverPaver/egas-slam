#!/usr/bin/env python3
"""Compile and run a small MiniSlamGraph regression for robust loop edges."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CPP_SOURCE = r"""
#include "MiniSlamGraphLib/GraphEdgeSE3.h"
#include "MiniSlamGraphLib/GraphNodeSE3.h"
#include "MiniSlamGraphLib/LevenbergMarquardtMethod.h"
#include "MiniSlamGraphLib/PoseGraph.h"
#include "MiniSlamGraphLib/SlamGraphErrorFunction.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace MiniSlamGraph;

static ORUtils::Matrix4<float> translation_matrix(float x)
{
    ORUtils::Matrix4<float> m;
    m.setIdentity();
    m.setTranslate(ORUtils::Vector3<float>(x, 0.0f, 0.0f));
    return m;
}

static void add_node(PoseGraph &graph, int id, float local_to_global_x, bool fixed)
{
    ORUtils::Matrix4<float> global_to_local = translation_matrix(-local_to_global_x);
    GraphNodeSE3 *node = new GraphNodeSE3(ORUtils::SE3Pose(global_to_local));
    node->setId(id);
    node->setFixed(fixed);
    graph.addNode(node);
}

static void add_edge(PoseGraph &graph,
                     int from_id,
                     int to_id,
                     float from_local_to_to_local_x,
                     double information_weight,
                     bool robust,
                     const std::vector<double> &information_diagonal = {})
{
    GraphEdgeSE3 *edge = new GraphEdgeSE3();
    edge->setFromNodeId(from_id);
    edge->setToNodeId(to_id);
    edge->setMeasurementSE3(ORUtils::SE3Pose(translation_matrix(from_local_to_to_local_x)));
    edge->setInformationWeight(information_weight);
    edge->setInformationDiagonal(information_diagonal.empty()
                                     ? std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
                                     : information_diagonal);
    if (robust)
        edge->setRobustKernel(GraphEdge::ROBUST_CAUCHY, 0.1);
    graph.addEdge(edge);
}

struct SolveResult
{
    std::vector<double> x;
    double initial_error = 0.0;
    double final_error = 0.0;
    int solver_result = 0;
};

static SolveResult solve_chain(bool bad_loop, bool robust_loop)
{
    PoseGraph graph;
    add_node(graph, 0, 0.0f, true);
    add_node(graph, 1, 1.08f, false);
    add_node(graph, 2, 2.18f, false);
    add_node(graph, 3, 3.35f, false);

    add_edge(graph, 0, 1, -1.0f, 1.0, false);
    add_edge(graph, 1, 2, -1.0f, 1.0, false);
    add_edge(graph, 2, 3, -1.0f, 1.0, false);
    add_edge(graph, 0, 3, bad_loop ? -1.5f : -3.0f, bad_loop ? 30.0 : 1.0, robust_loop);

    graph.prepareEvaluations();
    SolveResult result;
    result.initial_error = graph.evaluateF();
    SlamGraphErrorFunction error_function(graph);
    SlamGraphErrorFunction::Parameters parameters(graph);
    result.solver_result = LevenbergMarquardtMethod::minimize(error_function, parameters);
    graph.setNodeIndex(parameters.getNodes());
    result.final_error = graph.evaluateF();

    result.x.resize(4);
    for (int id = 0; id < 4; ++id)
    {
        const GraphNodeSE3 *node = static_cast<const GraphNodeSE3 *>(graph.getNodeIndex().find(id)->second);
        const ORUtils::Matrix4<float> global_to_local = node->getPose().GetM();
        result.x[id] = -double(global_to_local.at(3, 0));
    }
    return result;
}

static std::vector<double> normalized_diagonal(double rotation_gate_q, double translation_gate_m, double confidence = 1.0)
{
    const double raw_rot = 1.0 / (rotation_gate_q * rotation_gate_q);
    const double raw_trans = 1.0 / (translation_gate_m * translation_gate_m);
    const double normalizer = std::sqrt(raw_rot * raw_trans);
    return {
        confidence * raw_rot / normalizer,
        confidence * raw_rot / normalizer,
        confidence * raw_rot / normalizer,
        confidence * raw_trans / normalizer,
        confidence * raw_trans / normalizer,
        confidence * raw_trans / normalizer,
    };
}

static std::vector<double> unnormalized_diagonal(double rotation_gate_q, double translation_gate_m)
{
    const double raw_rot = 1.0 / (rotation_gate_q * rotation_gate_q);
    const double raw_trans = 1.0 / (translation_gate_m * translation_gate_m);
    return {raw_rot, raw_rot, raw_rot, raw_trans, raw_trans, raw_trans};
}

static SolveResult solve_drifted_chain_with_loop(const std::vector<double> &sequential_diagonal,
                                                 const std::vector<double> &loop_diagonal)
{
    PoseGraph graph;
    add_node(graph, 0, 0.0f, true);
    add_node(graph, 1, 1.08f, false);
    add_node(graph, 2, 2.18f, false);
    add_node(graph, 3, 3.35f, false);

    add_edge(graph, 0, 1, -1.08f, 1.0, false, sequential_diagonal);
    add_edge(graph, 1, 2, -1.10f, 1.0, false, sequential_diagonal);
    add_edge(graph, 2, 3, -1.17f, 1.0, false, sequential_diagonal);
    add_edge(graph, 0, 3, -3.0f, 1.0, false, loop_diagonal);

    graph.prepareEvaluations();
    SolveResult result;
    result.initial_error = graph.evaluateF();
    SlamGraphErrorFunction error_function(graph);
    SlamGraphErrorFunction::Parameters parameters(graph);
    result.solver_result = LevenbergMarquardtMethod::minimize(error_function, parameters);
    graph.setNodeIndex(parameters.getNodes());
    result.final_error = graph.evaluateF();

    result.x.resize(4);
    for (int id = 0; id < 4; ++id)
    {
        const GraphNodeSE3 *node = static_cast<const GraphNodeSE3 *>(graph.getNodeIndex().find(id)->second);
        const ORUtils::Matrix4<float> global_to_local = node->getPose().GetM();
        result.x[id] = -double(global_to_local.at(3, 0));
    }
    return result;
}

static double max_abs_error(const std::vector<double> &x)
{
    double ret = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
        ret = std::max(ret, std::fabs(x[i] - double(i)));
    return ret;
}

static void assert_true(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int main()
{
    SolveResult good_scalar = solve_chain(false, false);
    SolveResult good_robust = solve_chain(false, true);
    SolveResult bad_scalar = solve_chain(true, false);
    SolveResult bad_robust = solve_chain(true, true);

    const double sequential_translation_gate_m = 0.008;
    const double loop_translation_gate_m = 0.25;
    constexpr double kPi = 3.14159265358979323846;
    const double sequential_rotation_gate_q = std::sin((0.75 * kPi / 180.0) * 0.5);
    const double loop_rotation_gate_q = std::sin((20.0 * kPi / 180.0) * 0.5);
    SolveResult muted_diagonal = solve_drifted_chain_with_loop(
        unnormalized_diagonal(sequential_rotation_gate_q, sequential_translation_gate_m),
        unnormalized_diagonal(loop_rotation_gate_q, loop_translation_gate_m));
    SolveResult balanced_diagonal = solve_drifted_chain_with_loop(
        normalized_diagonal(sequential_rotation_gate_q, sequential_translation_gate_m),
        normalized_diagonal(loop_rotation_gate_q, loop_translation_gate_m));

    const double good_scalar_error = max_abs_error(good_scalar.x);
    const double good_robust_error = max_abs_error(good_robust.x);
    const double bad_scalar_error = max_abs_error(bad_scalar.x);
    const double bad_robust_error = max_abs_error(bad_robust.x);
    const double muted_loop_error = std::fabs(muted_diagonal.x[3] - 3.0);
    const double balanced_loop_error = std::fabs(balanced_diagonal.x[3] - 3.0);

    std::cout << "good_scalar_error=" << good_scalar_error
              << " good_robust_error=" << good_robust_error
              << " bad_scalar_error=" << bad_scalar_error
              << " bad_robust_error=" << bad_robust_error
              << " bad_scalar_final_error=" << bad_scalar.final_error
              << " bad_robust_final_error=" << bad_robust.final_error
              << " muted_loop_error=" << muted_loop_error
              << " balanced_loop_error=" << balanced_loop_error
              << std::endl;

    assert_true(good_scalar_error < 1e-3, "Scalar solve failed on a consistent loop");
    assert_true(good_robust_error < 1e-3, "Robust solve failed on a consistent loop");
    assert_true(bad_robust_error < bad_scalar_error * 0.5,
                "Robust loop kernel did not reduce damage from a bad loop enough");
    assert_true(balanced_loop_error < muted_loop_error * 0.5,
                "Normalized diagonal policy still mutes a geometrically consistent loop");
    return 0;
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "build_compact")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    minislam = build_dir / "InfiniTAM/MiniSlamGraphLib/libMiniSlamGraphLib.a"
    orutils = build_dir / "InfiniTAM/ORUtils/libORUtils.a"
    if not minislam.exists() or not orutils.exists():
        raise FileNotFoundError(
            f"MiniSlamGraph/ORUtils libraries not found under {build_dir}; build slam_trainer first"
        )

    with tempfile.TemporaryDirectory(prefix="gps_slam_pose_graph_regression_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "pose_graph_solver_regression.cpp"
        binary = tmp_path / "pose_graph_solver_regression"
        source.write_text(CPP_SOURCE)
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            "-I",
            str(PROJECT_ROOT / "InfiniTAM"),
            "-I",
            str(PROJECT_ROOT / "InfiniTAM/MiniSlamGraphLib"),
            "-I",
            str(PROJECT_ROOT / "InfiniTAM/ORUtils"),
            str(source),
            str(minislam),
            str(orutils),
            "-o",
            str(binary),
        ]
        subprocess.run(compile_cmd, check=True, cwd=PROJECT_ROOT)
        subprocess.run([str(binary)], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
