#include <QtTest/QtTest>

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2023  Paragon<french.paragon@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "graphs/graphs.h"
#include "graphs/graph_cut.h"
#include "graphs/graph_flow.h"

#include <MultidimArrays/MultidimArrays.h>

#include <random>

using EdgeList = QVector<QPair<int, int>>;

struct WeightedEdge {
	int vert1;
	int vert2;
	float weight;
};

Q_DECLARE_METATYPE(WeightedEdge);

using WeightedEdgeList = QVector<WeightedEdge>;

Q_DECLARE_METATYPE(StereoVision::GraphProcessing::EdgeDirectedType);
Q_DECLARE_METATYPE(StereoVision::GraphProcessing::GraphMovingDirection);

class TestGraphs: public QObject
{

	Q_OBJECT

private Q_SLOTS:

	void initTestCase();

	void testGraphBuild_data();
	void testGraphBuild();

	void testGraphCutVerticesAccess_data();
	void testGraphCutVerticesAccess();

	void testMaxFlowMinCut_data();
	void testMaxFlowMinCut();

private:
	std::default_random_engine re;
};

void TestGraphs::initTestCase() {
	std::random_device rd;
	re.seed(rd());
}

void TestGraphs::testGraphBuild_data() {


	QTest::addColumn<int>("nVerts");
	QTest::addColumn<StereoVision::GraphProcessing::EdgeDirectedType>("directedGraph");
	QTest::addColumn<QVector<QPair<int, int>>>("edges");
	QTest::addColumn<QVector<QPair<int, int>>>("nonExistingEdges2Test");

	QTest::newRow("2 Vertices undirected") << 2 << StereoVision::GraphProcessing::EdgeDirectedType::UndirectedEdges
										   << EdgeList{{0,1}} << EdgeList{};
	QTest::newRow("2 Vertices directed") << 2 << StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges
										 << EdgeList{{0,1}} << EdgeList{{1,0}};
}
void TestGraphs::testGraphBuild() {


	using UndirectedGraph = StereoVision::GraphProcessing::Graph<StereoVision::GraphProcessing::EdgeDirectedType::UndirectedEdges>;
	using DirectedGraph = StereoVision::GraphProcessing::Graph<StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges>;

	QFETCH(int, nVerts);
	QFETCH(StereoVision::GraphProcessing::EdgeDirectedType, directedGraph);
	QFETCH(EdgeList, edges);
	QFETCH(EdgeList, nonExistingEdges2Test);

	if (directedGraph == StereoVision::GraphProcessing::EdgeDirectedType::UndirectedEdges) {
		UndirectedGraph graph(nVerts);

		for (QPair<int, int> const& edge : edges) {
			graph.linkVertices(edge.first, edge.second);
		}

		for (QPair<int, int> const& edge : edges) {
			QVERIFY2(graph.edgeBetweenVerticesId(edge.first, edge.second).has_value(), "Missing edge that has been inserted");
			QVERIFY2(graph.edgeBetweenVerticesId(edge.second, edge.first).has_value(), "Missing edge (reverse order should not matter for undirected graph)");
		}

		for (QPair<int, int> const& edge : nonExistingEdges2Test) {
			QVERIFY2(!graph.edgeBetweenVerticesId(edge.first, edge.second).has_value(), "Non existing edge has been found");
		}

	} else if (directedGraph == StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges) {

		DirectedGraph graph(nVerts);

		for (QPair<int, int> const& edge : edges) {
			graph.linkVertices(edge.first, edge.second);
		}

		for (QPair<int, int> const& edge : edges) {
			QVERIFY2(graph.edgeBetweenVerticesId(edge.first, edge.second).has_value(), "Missing edge that has been inserted");
		}

		for (QPair<int, int> const& edge : nonExistingEdges2Test) {
			QVERIFY2(!graph.edgeBetweenVerticesId(edge.first, edge.second).has_value(), "Non existing edge has been found");
		}
	}

}



void TestGraphs::testGraphCutVerticesAccess_data() {

	constexpr StereoVision::GraphProcessing::GraphMovingDirection Both = StereoVision::GraphProcessing::GraphMovingDirection::Both;
	constexpr StereoVision::GraphProcessing::GraphMovingDirection Forward = StereoVision::GraphProcessing::GraphMovingDirection::Forward;
	constexpr StereoVision::GraphProcessing::GraphMovingDirection Backward = StereoVision::GraphProcessing::GraphMovingDirection::Backward;

	QTest::addColumn<int>("nVerts");
	QTest::addColumn<StereoVision::GraphProcessing::EdgeDirectedType>("directedGraph");
	QTest::addColumn<EdgeList>("edges");
	QTest::addColumn<QVector<int>>("edges2cut");
	QTest::addColumn<StereoVision::GraphProcessing::GraphMovingDirection>("searchDirection");
	QTest::addColumn<int>("startVertex");
	QTest::addColumn<QVector<int>>("reachableVertices");


	QTest::newRow("4 Vertices uncut directed sink") << 4 << StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges
										   << EdgeList{{0,1}, {1,2}, {2,3}} << QVector<int>{} << Both << 3 << QVector<int>{0,1,2};

	QTest::newRow("4 Vertices middle cut undirected source") << 4 << StereoVision::GraphProcessing::EdgeDirectedType::UndirectedEdges
										   << EdgeList{{0,1}, {1,2}, {2,3}} << QVector<int>{1} << Backward << 0 << QVector<int>{1};

	QTest::newRow("4 Vertices middle cut directed source") << 4 << StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges
										   << EdgeList{{0,1}, {1,2}, {2,3}} << QVector<int>{1} << Forward << 0 << QVector<int>{1};
	QTest::newRow("4 Vertices middle cut directed sink") << 4 << StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges
										   << EdgeList{{0,1}, {1,2}, {2,3}} << QVector<int>{1} << Backward << 3 << QVector<int>{2};

}
void TestGraphs::testGraphCutVerticesAccess() {

	using UndirectedGraph = StereoVision::GraphProcessing::Graph<StereoVision::GraphProcessing::EdgeDirectedType::UndirectedEdges>;
	using DirectedGraph = StereoVision::GraphProcessing::Graph<StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges>;

	constexpr StereoVision::GraphProcessing::GraphMovingDirection Both = StereoVision::GraphProcessing::GraphMovingDirection::Both;
	constexpr StereoVision::GraphProcessing::GraphMovingDirection Forward = StereoVision::GraphProcessing::GraphMovingDirection::Forward;
	constexpr StereoVision::GraphProcessing::GraphMovingDirection Backward = StereoVision::GraphProcessing::GraphMovingDirection::Backward;

	QFETCH(int, nVerts);
	QFETCH(StereoVision::GraphProcessing::EdgeDirectedType, directedGraph);
	QFETCH(EdgeList, edges);
	QFETCH(QVector<int>, edges2cut);
	QFETCH(StereoVision::GraphProcessing::GraphMovingDirection, searchDirection);
	QFETCH(int, startVertex);
	QFETCH(QVector<int>, reachableVertices);

	std::vector<int> cuttedEdgesId(edges2cut.begin(), edges2cut.end());

	if (directedGraph == StereoVision::GraphProcessing::EdgeDirectedType::UndirectedEdges) {
		UndirectedGraph graph(nVerts);

		for (QPair<int, int> const& edge : edges) {
			graph.linkVertices(edge.first, edge.second);
		}

		std::vector<int> reachedVertices = StereoVision::GraphProcessing::reachableVerticesInCut<UndirectedGraph, Both>(graph, cuttedEdgesId, startVertex);

		std::sort(reachableVertices.begin(), reachableVertices.end());
		std::sort(reachedVertices.begin(), reachedVertices.end());

		QCOMPARE(reachedVertices.size(), reachableVertices.size());

		for (size_t i = 0; i < reachedVertices.size(); i++) {
			QCOMPARE(reachedVertices[i], reachableVertices[i]);
		}


	} else if (directedGraph == StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges) {

		DirectedGraph graph(nVerts);

		for (QPair<int, int> const& edge : edges) {
			graph.linkVertices(edge.first, edge.second);
		}

		std::vector<int> reachedVertices;

		if (searchDirection == Both) {
			reachedVertices = StereoVision::GraphProcessing::reachableVerticesInCut<DirectedGraph, Both>(graph, cuttedEdgesId, startVertex);
		} else if (searchDirection == Forward) {
			reachedVertices = StereoVision::GraphProcessing::reachableVerticesInCut<DirectedGraph, Forward>(graph, cuttedEdgesId, startVertex);
		} else if (searchDirection == Backward) {
			reachedVertices = StereoVision::GraphProcessing::reachableVerticesInCut<DirectedGraph, Backward>(graph, cuttedEdgesId, startVertex);
		}

		std::sort(reachableVertices.begin(), reachableVertices.end());
		std::sort(reachedVertices.begin(), reachedVertices.end());

		QCOMPARE(reachedVertices.size(), reachableVertices.size());

		for (size_t i = 0; i < reachedVertices.size(); i++) {
			QCOMPARE(reachedVertices[i], reachableVertices[i]);
		}
	}

}

void TestGraphs::testMaxFlowMinCut_data() {

	QTest::addColumn<int>("nVerts");
	QTest::addColumn<WeightedEdgeList>("edges");
	QTest::addColumn<int>("sourceVertexId");
	QTest::addColumn<int>("targetVertexId");
	QTest::addColumn<float>("maxFlow");
	QTest::addColumn<QVector<int>>("minCut");

	float largeWeight = 2.0;
	float smallWeight = 1.0;

	WeightedEdgeList edgesLinearGraph{{0,1,largeWeight}, {1,2, smallWeight}, {2,3, largeWeight}};
	QTest::newRow("Linear graph") << 4 << edgesLinearGraph << 0 << 3 << smallWeight << QVector<int>{1};

	WeightedEdgeList edgesTwoPathGraph{{0,1,2*largeWeight}, {1,2, largeWeight},
									   {1,3, smallWeight}, {2,4, smallWeight},
									   {3,5, largeWeight}, {4,5, largeWeight}};
	QTest::newRow("Two path graph") << 6 << edgesTwoPathGraph << 0 << 5 << (2*smallWeight) << QVector<int>{2,3};

	//an example copied from wikipedia [https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm]
	WeightedEdgeList wikipediaExample1{{0,1, 3*smallWeight}, {0,3, 3*smallWeight},
									   {1,2, 4*smallWeight},
									   {2,0, 3*smallWeight}, {2,3, smallWeight}, {2,4, 2*smallWeight},
									   {3,4, 2*smallWeight}, {3,5, 6*smallWeight},
									   {4,1, smallWeight}, {4,6, smallWeight},
									   {5,6, 9*smallWeight}};
	QTest::newRow("Wikipedia example 1") << 7 << wikipediaExample1 << 0 << 6 << (5*smallWeight) << QVector<int>{1,4,9};
}
void TestGraphs::testMaxFlowMinCut() {

	using DirectedGraph = StereoVision::GraphProcessing::Graph<StereoVision::GraphProcessing::EdgeDirectedType::DirectedEdges, void, float>;

	QFETCH(int, nVerts);
	QFETCH(WeightedEdgeList, edges);
	QFETCH(int, sourceVertexId);
	QFETCH(int, targetVertexId);
	QFETCH(float, maxFlow);
	QFETCH(QVector<int>, minCut);

	DirectedGraph graph(nVerts);

	for (WeightedEdge const& edge : edges) {
		graph.linkVertices(edge.vert1, edge.vert2, edge.weight);
	}

	auto maxFlowMinCut = StereoVision::GraphProcessing::maxFlowMinCut(graph, sourceVertexId, targetVertexId);

	QCOMPARE(maxFlowMinCut.maxFlow, maxFlow);

	std::sort(minCut.begin(), minCut.end());
	std::sort(maxFlowMinCut.minCutEdgesIdxs.begin(), maxFlowMinCut.minCutEdgesIdxs.end());

	QCOMPARE(maxFlowMinCut.minCutEdgesIdxs.size(), minCut.size());

	for (int i = 0; i < minCut.size(); i++) {
		QCOMPARE(maxFlowMinCut.minCutEdgesIdxs[i], minCut[i]);
	}
}

QTEST_MAIN(TestGraphs)
#include "testGraphs.moc"
