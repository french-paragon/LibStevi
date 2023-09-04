#ifndef LIBSTEVI_ASSIGNEMENT_PROBLEMS_H
#define LIBSTEVI_ASSIGNEMENT_PROBLEMS_H

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

#include <eigen3/Eigen/Core>


namespace StereoVision {
namespace Optimization {

/*!
 * \brief optimalAssignement compute an optimal assignement between two finite sets, given a cost matrix
 * \param Costs the costs of assigning elements to one another, as a matrix.
 * \return a list of pairs, matching elements of the rows on the matrix to the elements of the columns of the matrix.
 *
 * Current implementation use the hungarian method.
 */
template<typename T>
std::vector<std::array<int, 2>> optimalAssignement(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const& Costs) {

    //TODO: implement cubic complexity graph based algorithm

    using CostT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
    using ColVec = Eigen::Vector<T, Eigen::Dynamic>;

    int o_n = Costs.rows();
    int o_m = Costs.cols();

    int maxS = std::max(o_n,o_m);

    CostT costs;
    costs.resize(maxS, maxS);

    costs.block(0,0,o_n,o_m) = Costs;

    if (o_n < o_m) {
        costs.block(o_n,0,maxS-o_n,o_m) = CostT::Zero(maxS-o_n,o_m);
    }

    if (o_m < o_n) {
        costs.block(0,o_m,o_n,maxS-o_m) = CostT::Zero(o_n,maxS-o_m);
    }

    int n = costs.rows();
    int m = costs.cols();

    assert(n <= m);

    //adding or subtracting a constant to a row or column leave the optimal assignement invariant (as long as the matrix is squared.

    //First, remove the minimal cost for each row
    ColVec rowMin = costs.rowwise().minCoeff();
    costs.colwise() -= rowMin;

    //Second remove the minimal cost for each column
    RowVec colMin = costs.colwise().minCoeff();
    costs.rowwise() -= colMin;

    //try to do an assignement
    std::vector<int> stared0InRows(n);
    std::fill(stared0InRows.begin(), stared0InRows.end(), -1);

    std::vector<int> stared0InCols(m);
    std::fill(stared0InCols.begin(), stared0InCols.end(), -1);

    int nStared = 0;

    //Try to find a zero to star in each row
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {

            if (costs(i,j) == 0) {
                if (stared0InCols[j] < 0) {
                    stared0InRows[i] = j;
                    stared0InCols[j] = i;
                    nStared++;
                    break;
                }
            }
        }
    }

    //increase the number of stared 0 by 1 each iteration (or ensure it can be on the next iteration
    while (nStared < n) {

        std::vector<uint8_t> coveredRows(n);
        std::fill(coveredRows.begin(), coveredRows.end(), false);

        std::vector<uint8_t> coveredCols(m);
        for (int i = 0; i < m; i++) {
            coveredCols[i] = stared0InCols[i] >= 0;
        }

        bool starAdded = false;

        std::vector<int> primed0InRows(n);
        std::fill(primed0InRows.begin(), primed0InRows.end(), -1);

        for (int i = 0; i < n; i++) {

            if (coveredRows[i]) { //we want a non covered 0
                continue;
            }

            for (int j = 0; j < m; j++) {

                if (coveredCols[j]) { //we want a non covered 0
                    continue;
                }

                if (costs(i,j) != 0) {
                    continue;
                }

                if (stared0InRows[i] == j) {
                    continue;
                }

                primed0InRows[i] = j;

                //if there is a stared 0 in the row, cover the row;
                if (stared0InRows[i] >= 0) {
                    coveredCols[stared0InRows[i]] = false;
                    coveredRows[i] = true;

                    //restart the search
                    i = -1;
                    break;
                }

                //else, start the chain to add a stared 0

                std::vector<std::array<int,2>> toUnstar;
                std::vector<std::array<int,2>> toStar;

                int currentRow = i;
                int currentCol = j;

                int nextStarRow = stared0InCols[currentCol];
                int nextPrimeCol;

                while(true) {
                    toStar.push_back({currentRow, currentCol});

                    toUnstar.push_back({nextStarRow, currentCol});

                    nextPrimeCol = primed0InRows[nextStarRow];

                    currentRow = nextStarRow;
                    currentCol = nextPrimeCol;

                    nextStarRow = stared0InCols[nextPrimeCol];

                    if (nextStarRow < 0) {
                        toStar.push_back({currentRow, currentCol});
                        break;
                    }
                }

                assert(toUnstar.size()+1 == toStar.size());

                for (std::array<int,2> & unstar : toUnstar) {

                    stared0InRows[unstar[0]] = -1;
                    stared0InCols[unstar[1]] = -1;

                }

                for (std::array<int,2> & star : toStar) {

                    stared0InRows[star[0]] = star[1];
                    stared0InCols[star[1]] = star[0];

                }

                nStared++;
                starAdded = true;
            }

            if (starAdded) {
                break;
            }
        }

        //no star added, we need to create additional 0s
        if (!starAdded) {

            T minCost = costs.maxCoeff();

            for (int i = 0; i < n; i++) {

                if (coveredRows[i]) {
                    continue;
                }

                for (int j = 0; j < m; j++) {

                    if (coveredCols[j]) {
                        continue;
                    }

                    T c = costs(i,j);

                    if (minCost > c) {
                        minCost = c;
                    }
                }
            }

            //subtract the min uncovered cost to all uncovered rows and add it to all covered columns.
            for (int i = 0; i < n; i++) {

                for (int j = 0; j < m; j++) {

                    if (coveredRows[i] and coveredCols[j]) {
                        assert(stared0InRows[i] != j and stared0InCols[j] != i);
                        costs(i,j) += minCost;
                    } else if (!coveredRows[i] and !coveredCols[j]) {
                        assert(stared0InRows[i] != j and stared0InCols[j] != i);
                        costs(i,j) -= minCost;
                    }
                }
            }
            //stared 0 should remain unnafected

        }

        //now either a new star has been added, or a new 0 cost candidate has been created.
    }

    std::vector<std::array<int, 2>> ret;
    ret.reserve(n);

    for (int i = 0; i < n; i++) {
        if (i < o_n and stared0InRows[i] < o_m) {
            ret.push_back({i,stared0InRows[i]});
        }
    }

    return ret;

}

}
}

#endif // LIBSTEVI_ASSIGNEMENT_PROBLEMS_H
