#ifndef DOUBLYCONNECTEDEDGELISTS_H
#define DOUBLYCONNECTEDEDGELISTS_H
/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2024  Paragon<french.paragon@gmail.com>

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

#include "./graphs.h"

#include <cassert>

namespace StereoVision {

namespace GraphProcessing {

enum DCELFaceType {
    Triangle,
    NGon
};

/*
 * Doubly connected edge lists (DCEL) are efficient data structure to store and process planar graphs.
 *
 * This file contain some implementation of DCEL data structures used to store general planar graphs, meshes and other related
 * data structures.
 */

class DCELHalfEdge {
public:

    DCELHalfEdge() :
        _vertex1_id(-1),
        _vertex2_id(-1),
        _previous_id(-1),
        _next_id(-1),
        _twin_id(-1),
        _faceId(-1)
    {

    }

    DCELHalfEdge(int v1_id, int v2_id, int previousId = -1, int nextId = -1, int twinId = -1) :
        _vertex1_id(v1_id),
        _vertex2_id(v2_id),
        _previous_id(previousId),
        _next_id(nextId),
        _twin_id(twinId),
        _faceId(-1)
    {

    }

    inline bool isValid() const {
        return _vertex1_id >= 0 and _vertex2_id >= 0 and _vertex1_id != _vertex2_id;
    }

    inline int vertex1() const {
        return _vertex1_id;
    }

    inline int vertex2() const {
        return _vertex2_id;
    }

    inline int vertex(VertexEdgePosition vType) const {
        return (vType == SourceVertex) ? _vertex1_id : _vertex2_id;
    }

    inline int previousHalfEdge() const {
        return _previous_id;
    }

    inline int nextHalfEdge() const {
        return _next_id;
    }

    inline int twinHalfEdge() const {
        return _next_id;
    }

    inline void setTwinHalfEdge(int id) {
        _next_id = id;
    }

    /*!
     * \brief faceId get the face the half edge belongs to.
     * \return The index of the face.
     */
    inline int faceId() const {
        return _faceId;
    }

protected:
    int _vertex1_id;
    int _vertex2_id;

    int _previous_id;
    int _next_id;
    int _twin_id;

    int _faceId;

    template<DCELFaceType E_T, typename VD_T>
    friend class GenericDoublyConnectedEdgeList;
};

template<DCELFaceType F_T, typename VD_T = void>
class GenericDoublyConnectedEdgeList {
public:

    static const EdgeDirectedType edgeType = DirectedEdges; //The edges have to be directed in a DCEL

    //typedefs provided to be compatible with a generic graph concept
    using VertexT = GraphVertex<VD_T>;
    using EdgeT = DCELHalfEdge;

    using VertexDataType = VD_T;
    using EdgeWeightType = void;

    using FaceVertexList = std::conditional_t<F_T == Triangle, std::array<int,3>, std::vector<int>>;
    using FaceEdgeList = FaceVertexList;

    struct FullFace {
        FaceVertexList vertices;
        FaceEdgeList edges;
    };

    struct VertexLocalEdgeSequence {
        int enteringEdge;
        int leavingEdge;
    };

    GenericDoublyConnectedEdgeList() {

    }

    GenericDoublyConnectedEdgeList(int nVertex) {
        _vertices.reserve(nVertex);

        for (int i = 0; i < nVertex; i++) {
            _vertices.emplace_back(i);
        }
    }

    template<typename pV_T>
    GenericDoublyConnectedEdgeList(int nVertex, std::enable_if_t<!std::is_void_v<pV_T> and std::is_same_v<pV_T, VD_T>, VD_T> val = VD_T()) {
        _vertices.reserve(nVertex);

        for (int i = 0; i < _vertices.size(); i++) {
            _vertices.emplace_back(i,val);
        }
    }

    inline int nVertices() const {
        return _vertices.size();
    }


    inline int nEdges() const {
        return _edges.size();
    }

    inline VertexT const& vertex(int i) const {
        return _vertices[i];
    }

    inline std::conditional_t<std::is_void_v<VD_T>, void, std::conditional_t<std::is_void_v<VD_T>, int, VD_T>&> vertexData(int i) {
        return _vertices[i].data();
    }

    inline EdgeT const& edge(int i) const {
        return _edges[i];
    }

    /*!
     * \brief edgeBetweenVerticesId
     * \param idv1
     * \param idv2
     * \return the index of one edge going from the first vertex to the second vertex.
     *
     * This function use the specific structure of the DCEL to find the edge connecting the first vertex to the second one.
     * Each vertex has a referecence to one of its edge. If this edge points to the second vertex it is returned.
     * If this edge do not point toward the second vertex, then its twin is followed to get the next edge leaving the first vertex.
     * Once the second vertex is found, the corresponding edge is returned.
     * If the initial edge is found again, no edge is returned.
     *
     * The worst running time of this function is O(number of outgoing edges from the vertex)
     */
    inline std::optional<int> edgeBetweenVerticesId(int idv1, int idv2) const {

        int initialEdgeId = _verticesOutEdge[idv1];
        int currentEdgeId = initialEdgeId;

        do {
            if (currentEdgeId < 0 or currentEdgeId >= _edges.size()) {
                break;
            }

            if (_edges[currentEdgeId].vertex2() == idv2) {
                return currentEdgeId;
            }

            currentEdgeId = _edges[currentEdgeId].twinHalfEdge();

            if (currentEdgeId < 0 or currentEdgeId >= _edges.size()) {
                break;
            }

            currentEdgeId = _edges[currentEdgeId].nextHalfEdge();


        } while (currentEdgeId != initialEdgeId);

        return std::nullopt;
    }

    std::optional<VertexLocalEdgeSequence> getExteriorEdgeForVertex(int vertexId) const {

        int initialEdgeId = _verticesOutEdge[vertexId];
        int currentEdgeId = initialEdgeId;

        do {
            if (currentEdgeId < 0 or currentEdgeId >= _edges.size()) {
                break;
            }

            if (_edges[currentEdgeId].faceId() < 0) {
                return {_edges[currentEdgeId].previousHalfEdge(), currentEdgeId};
            }

            currentEdgeId = _edges[currentEdgeId].twinHalfEdge();

            if (currentEdgeId < 0 or currentEdgeId >= _edges.size()) {
                break;
            }

            if (_edges[currentEdgeId].faceId() < 0) {
                return {currentEdgeId, _edges[currentEdgeId].nextHalfEdge()};
            }

            currentEdgeId = _edges[currentEdgeId].nextHalfEdge();


        } while (currentEdgeId != initialEdgeId);

        return std::nullopt;
    }

    /*!
     * \brief makeFace create a face in the graph
     * \param vertices the list of vertices to put in the face
     * \return the face id, or -1 if the face could not be created
     */
    inline int makeFace(FaceVertexList const& vertices) {

        int faceId = _faces.size();

        FaceEdgeList emptyList = vertices; //copy the container to bypass constructor syntax difference betweeen fixed and dynamic size
        for (int & idx : emptyList) {
            idx = -1;
        }

        FaceEdgeList edges2Integrate = emptyList; //edges from the exterior wich will be assimilated in the face

        FaceEdgeList externalEdges2ReconnectBack = emptyList; //edge from the exterior leaving the vertex which need to be reconnected.
        FaceEdgeList external2ReconnectFront = emptyList; // edges from the exterior entering the vertex which need to be reconnected.
        FaceEdgeList edges4BackConnect = emptyList; //edges from the face that will be used for the reconnection of the leaving external edge.
        FaceEdgeList edges4FrontConnect = emptyList; //edges from the face that will be used for the reconnection of the entering external edge.

        //check the vertices, return an error before changing the structure if creating the face is not possible
        for (int i = 0; i < vertices.size(); i++) {
            int currentVertex = vertices[i];
            int nextVertex = vertices[(i+1)%vertices.size()];

            edges2Integrate[i] = edgeBetweenVerticesId(currentVertex, nextVertex);

            if (edges2Integrate[i] >= 0) {
                if (_edges[edges2Integrate[i]].faceId() >= 0) {
                    return -1; //cannot create a face if an half edge is already in another face (not the exterior face)
                }
            }
        }

        FaceEdgeList internal2ReconnectFront = emptyList;
        FaceEdgeList internalEdges2ReconnectBack = emptyList;
        int faceStartEdge = -1;

        //setup the edges, one by one
        for (int i = 0; i < vertices.size(); i++) {

            int currentVertex = vertices[i];
            int nextVertex = vertices[(i+1)%vertices.size()];

            if (edges2Integrate[i] >= 0) {
                //we disconnect an edge from the exterior
                _edges[edges2Integrate[i]]._faceId = faceId;
                //we need to mark the disconnected edges for reconnection
                external2ReconnectFront[i] = _edges[edges2Integrate[i]].previousHalfEdge();
                externalEdges2ReconnectBack[(i+1)%vertices.size()] = _edges[edges2Integrate[i]].nextHalfEdge();

                internalEdges2ReconnectBack[i] = edges2Integrate[i];
                internal2ReconnectFront[i] = edges2Integrate[i];

                if (i == 0) {
                    faceStartEdge = edges2Integrate[i];
                }
            } else {
                //Need to create the edges
                int edgeId = _edges.size();
                int twinId = _edges.size()+1;

                int temporaryPreviousId = -1;
                int temporaryNextId = -1;

                _edges.emplace_back(currentVertex, nextVertex, temporaryPreviousId, temporaryNextId, twinId);
                _edges.emplace_back(nextVertex, currentVertex, temporaryPreviousId, temporaryNextId, edgeId);

                _edges[edgeId]._faceId = faceId;
                _edges[twinId]._faceId = -1;

                internalEdges2ReconnectBack[i] = edgeId;
                internal2ReconnectFront[(i+1)%vertices.size()] = edgeId;

                external2ReconnectFront[i] = twinId;
                externalEdges2ReconnectBack[(i+1)%vertices.size()] = twinId;

                if (i == 0) {
                    faceStartEdge = edgeId;
                }
            }

        }


        //reconnect the edges
        for (int i = 0; i < vertices.size(); i++) {

            int currentVertex = vertices[i];
            int nextVertex = vertices[(i+1)%vertices.size()];

            assert((external2ReconnectFront[i] < 0 and externalEdges2ReconnectBack[i] < 0) or
                   (external2ReconnectFront[i] >= 0 and externalEdges2ReconnectBack[i] >= 0));

            if (external2ReconnectFront[i] >= 0 and externalEdges2ReconnectBack[i] >= 0) {
                _edges[external2ReconnectFront[i]]._next_id = externalEdges2ReconnectBack[i];
                _edges[externalEdges2ReconnectBack[i]]._previous_id = external2ReconnectFront[i];
            }

            assert((internal2ReconnectFront[i] < 0 and internalEdges2ReconnectBack[i] < 0) or
                   (internal2ReconnectFront[i] >= 0 and internalEdges2ReconnectBack[i] >= 0));

            if (internal2ReconnectFront[i] >= 0 and internalEdges2ReconnectBack[i] >= 0) {
                _edges[internal2ReconnectFront[i]]._next_id = internalEdges2ReconnectBack[i];
                _edges[internalEdges2ReconnectBack[i]]._previous_id = internal2ReconnectFront[i];
            }
        }

        _faces.push_back(faceStartEdge);
        return faceId;

    }

protected:
    std::vector<VertexT> _vertices;
    std::vector<int> _verticesOutEdge;
    std::vector<EdgeT> _edges;
    std::vector<int> _faces; //! points to one half edge in the face

};



} // namespace GraphProcessing

} // namespace StereoVision

#endif // DOUBLYCONNECTEDEDGELISTS_H
