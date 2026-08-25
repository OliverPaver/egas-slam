// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "GraphEdge.h"

#include <cmath>
#include <stdexcept>

using namespace MiniSlamGraph;

static void jacobianToHessian_diagonalpart(const double *residual, const double *jacobian, const std::vector<double> &weights, int dimMeasure, int numPara, double *Gblock, double *Hblock)
{
	for (int para = 0; para < numPara; ++para) {
		Gblock[para] = 0.0f;
		for (int idx = 0; idx < dimMeasure; ++idx) {
			//Gblock[row] += jacobian[row*dimMeasure+idx] * residual[idx];
			Gblock[para] += jacobian[idx*numPara + para] * residual[idx] * weights[idx];
		}

		for (int para2 = para; para2 < numPara; ++para2) {
			Hblock[para*numPara + para2] = 0.0f;
			//for (int idx = 0; idx < dimMeasure; ++idx) Hblock[row*numPara+col] += jacobian[row*dimMeasure+idx] * jacobian[col*dimMeasure+idx];
			for (int idx = 0; idx < dimMeasure; ++idx) Hblock[para*numPara + para2] += jacobian[idx*numPara + para] * jacobian[idx*numPara + para2] * weights[idx];
		}
	}
	for (int row = 0; row < numPara; ++row) {
		for (int col = 0; col < row; ++col) {
			Hblock[row*numPara + col] = Hblock[col*numPara + row];
		}
	}
}

static void jacobianToHessian_offdiagonal(const double *jacobian_from, const double *jacobian_to, const std::vector<double> &weights, int dimMeasure, int numPara_from, int numPara_to, double *Hblock)
{
	for (int row = 0; row < numPara_from; ++row) {
		for (int col = 0; col < numPara_to; ++col) {
			Hblock[row*numPara_to + col] = 0.0f;
			//for (int idx = 0; idx < dimMeasure; ++idx) Hblock[row*numPara_to+col] += jacobian_from[row*dimMeasure+idx] * jacobian_to[col*dimMeasure+idx];
			for (int idx = 0; idx < dimMeasure; ++idx) Hblock[row*numPara_to + col] += jacobian_from[idx*numPara_from + row] * jacobian_to[idx*numPara_to + col] * weights[idx];
		}
	}
}

static std::vector<double> effectiveInformationWeights(const GraphEdge &edge, int dim)
{
	std::vector<double> weights(dim, edge.getInformationWeight());
	const std::vector<double> &diagonal = edge.getInformationDiagonal();
	if (diagonal.empty()) return weights;
	if (diagonal.size() != static_cast<size_t>(dim))
		throw std::runtime_error("MiniSlamGraph edge information diagonal has the wrong dimension");
	for (int i = 0; i < dim; ++i) weights[i] *= diagonal[i];
	return weights;
}

static double weightedResidualSquaredNorm(const double *residual, const std::vector<double> &weights, int dim)
{
	double ret = 0.0;
	for (int i = 0; i < dim; ++i) ret += weights[i] * residual[i] * residual[i];
	return ret;
}

static double robustCost(GraphEdge::RobustKernelType type, double scale, double weightedSquaredNorm)
{
	if (!std::isfinite(weightedSquaredNorm) || weightedSquaredNorm < 0.0)
		return weightedSquaredNorm;
	const double safeScale = (std::isfinite(scale) && scale > 0.0) ? scale : 1.0;
	if (type == GraphEdge::ROBUST_NONE)
		return 0.5 * weightedSquaredNorm;

	const double norm = std::sqrt(weightedSquaredNorm);
	if (type == GraphEdge::ROBUST_HUBER)
	{
		if (norm <= safeScale)
			return 0.5 * weightedSquaredNorm;
		return safeScale * norm - 0.5 * safeScale * safeScale;
	}
	if (type == GraphEdge::ROBUST_CAUCHY)
		return 0.5 * safeScale * safeScale * std::log1p(weightedSquaredNorm / (safeScale * safeScale));
	return 0.5 * weightedSquaredNorm;
}

static double robustGradientScale(GraphEdge::RobustKernelType type, double scale, double weightedSquaredNorm)
{
	if (!std::isfinite(weightedSquaredNorm) || weightedSquaredNorm <= 0.0)
		return 1.0;
	const double safeScale = (std::isfinite(scale) && scale > 0.0) ? scale : 1.0;
	if (type == GraphEdge::ROBUST_HUBER)
	{
		const double norm = std::sqrt(weightedSquaredNorm);
		return norm <= safeScale ? 1.0 : safeScale / norm;
	}
	if (type == GraphEdge::ROBUST_CAUCHY)
		return 1.0 / (1.0 + weightedSquaredNorm / (safeScale * safeScale));
	return 1.0;
}

void GraphEdge::setInformationWeight(double weight)
{
	informationWeight = (std::isfinite(weight) && weight > 0.0) ? weight : 1.0;
}

void GraphEdge::setInformationDiagonal(const std::vector<double> &diagonal)
{
	const int dim = getMeasureDimensions();
	if (diagonal.size() != static_cast<size_t>(dim))
		throw std::invalid_argument("MiniSlamGraph edge information diagonal dimension mismatch");
	for (double value : diagonal)
	{
		if (!std::isfinite(value) || value <= 0.0)
			throw std::invalid_argument("MiniSlamGraph edge information diagonal must contain positive finite values");
	}
	informationDiagonal = diagonal;
}

void GraphEdge::setRobustKernel(RobustKernelType type, double scale)
{
	if (type != ROBUST_NONE && type != ROBUST_HUBER && type != ROBUST_CAUCHY)
		throw std::invalid_argument("Unsupported MiniSlamGraph robust kernel type");
	if (!std::isfinite(scale) || scale <= 0.0)
		throw std::invalid_argument("MiniSlamGraph robust kernel scale must be positive and finite");
	robustKernelType = type;
	robustKernelScale = scale;
}

double GraphEdge::computeError(const GraphEdge::NodeIndex & nodes) const
{
	int dim = getMeasureDimensions();
	std::vector<double> residual(dim);
	computeResidualVector(nodes, &(residual[0]));

	const std::vector<double> weights = effectiveInformationWeights(*this, dim);
	const double weightedSquaredNorm = weightedResidualSquaredNorm(&(residual[0]), weights, dim);
	return robustCost(robustKernelType, robustKernelScale, weightedSquaredNorm);
}

void GraphEdge::computeGradientAndHessian(const GraphEdge::NodeIndex & nodes, const ParameterIndex & index, VariableLengthVector & gradient, SparseBlockMatrix & hessian) const
{
	int id_from = fromNodeId();
	int id_to = toNodeId();
	int row_f = index.findIndex(id_from);
	int row_t = index.findIndex(id_to);
	//fprintf(stderr, "grad and hessian for edge from %i to %i (rows %i %i)\n", id_from, id_to, row_f, row_t);
	bool do_from = (row_f >= 0);
	bool do_to = (row_t >= 0);

	GraphNode *fromNode = nodes.find(id_from)->second;
	GraphNode *toNode = nodes.find(id_to)->second;
	int numPara_from = fromNode->numParameters();
	int numPara_to = toNode->numParameters();
	int dimMeasure = getMeasureDimensions();

	std::vector<double> residual(dimMeasure);
	std::vector<double> jacobian_from(dimMeasure*numPara_from);
	std::vector<double> jacobian_to(dimMeasure*numPara_to);

	computeResidualVector(nodes, &(residual[0]));
	if (do_from) computeJacobian(nodes, id_from, &(jacobian_from[0]));
	if (do_to) computeJacobian(nodes, id_to, &(jacobian_to[0]));

	std::vector<double> weights = effectiveInformationWeights(*this, dimMeasure);
	const double weightedSquaredNorm = weightedResidualSquaredNorm(&(residual[0]), weights, dimMeasure);
	const double robustScale = robustGradientScale(robustKernelType, robustKernelScale, weightedSquaredNorm);
	if (robustScale != 1.0)
	{
		for (double &weight : weights) weight *= robustScale;
	}

	// deal with "from" node
	if (do_from) {
		std::vector<double> Hblock_diag_f(numPara_from*numPara_from);
		std::vector<double> Gblock_f(numPara_from);
		jacobianToHessian_diagonalpart(&(residual[0]), &(jacobian_from[0]), weights, dimMeasure, numPara_from, &(Gblock_f[0]), &(Hblock_diag_f[0]));

		gradient.addData(row_f, numPara_from, &(Gblock_f[0]));
		hessian.addBlock(row_f, row_f, numPara_from, numPara_from, &(Hblock_diag_f[0]));
	}

	// deal with "to" node
	if (do_to) {
		std::vector<double> Hblock_diag_t(numPara_to*numPara_to);
		std::vector<double> Gblock_t(numPara_to);

		jacobianToHessian_diagonalpart(&(residual[0]), &(jacobian_to[0]), weights, dimMeasure, numPara_to, &(Gblock_t[0]), &(Hblock_diag_t[0]));

		gradient.addData(row_t, numPara_to, &(Gblock_t[0]));
		hessian.addBlock(row_t, row_t, numPara_to, numPara_to, &(Hblock_diag_t[0]));
	}

	// off diagonal part
	if (do_from && do_to) {
		std::vector<double> Hblock_off(numPara_from*numPara_to);
		jacobianToHessian_offdiagonal(&(jacobian_from[0]), &(jacobian_to[0]), weights, dimMeasure, numPara_from, numPara_to, &(Hblock_off[0]));

		hessian.addBlock(row_f, row_t, numPara_from, numPara_to, &(Hblock_off[0]));
		hessian.addBlockTranspose(row_t, row_f, numPara_to, numPara_from, &(Hblock_off[0]));
	}
}
