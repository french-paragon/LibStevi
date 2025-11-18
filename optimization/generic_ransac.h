#ifndef STEREO_VISION_GENERIC_RANSAC_H
#define STEREO_VISION_GENERIC_RANSAC_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2025  Paragon<french.paragon@gmail.com>

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

#include <vector>
#include <random>
#include <mutex>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace StereoVision {
namespace Optimization {

/*!
 * \brief The GenericRansac class is an helper class to build generic ransac optimisator
 *
 * The classe takes three template arguments, the measure, which has to be copy-constructible, the Model and optionally the sampling strategy.
 *
 * The model can be built from a collection of measures, and then gives an adequacy metric for all measures.
 * The model has to be copy constructible, constructible from a vector of measures and default constructible.
 * It is recommanded that move semantics be supported.
 * The model also require a method, error, taking as input a measurement and returning a type implicitly castable to double.
 *
 * If void is set as sampling strategy, measures are sampled uniformly at random.
 * Else, the sampling strategy is expected to be default constructible or copy constructible.
 * The sampling strategy should have an operator() taking as input a templated random engine type, an int number of measurements to generate, an int thread id and an int thread num.
 * The sampling strategy operator() should return a type that can be assigned to a std::vector<int>, which contains the idxs of the selected measurements.
 *
 */
template<typename Measure, typename Model, typename SamplingStrategy = void>
class GenericRansac {
public:

    template<typename MeasuresCollection>
    GenericRansac(MeasuresCollection && measures, int minimumMeasures, double inlierThreshold) :
        _measures(std::forward<MeasuresCollection>(measures)),
        _minimumMeasures(minimumMeasures),
        _threshold(inlierThreshold),
        _nInliers(0),
        _currentInliers(measures.size()),
        _currentModel(),
        _samplingStrategy()
    {
        std::default_random_engine re;

        _re.seed(re());
    }

    template<typename ST, typename OutT>
    using CheckedSamplerT = std::enable_if_t<std::is_same_v<std::remove_reference_t<ST>,std::remove_reference_t<SamplingStrategy>> and !std::is_same_v<SamplingStrategy,void>, OutT>;

    template<typename MeasuresCollection, typename SamplerT>
    GenericRansac(MeasuresCollection && measures,
                  SamplerT && sampler,
                  CheckedSamplerT<SamplerT,int> minimumMeasures,
                  double inlierThreshold) :
        _measures(std::forward<MeasuresCollection>(measures)),
        _minimumMeasures(minimumMeasures),
        _threshold(inlierThreshold),
        _nInliers(0),
        _currentInliers(measures.size()),
        _currentModel(),
        _samplingStrategy(std::forward<SamplerT>(sampler))
    {
        std::default_random_engine re;

        _re.seed(re());
    }

    inline int currentNInliers() const {
        return _nInliers;
    }

    inline Model const& currentModel() const {
        return _currentModel;
    }

    inline std::vector<bool> const& currentInliers() const {
        return _currentInliers;
    }

    /*!
     * \brief ransacIteration run one iterator of the ransac loop
     */
    inline void ransacIteration() {
        ransacIterationImpl(_re);
    }


    /*!
     * \brief ransacIterations perform a large number of ransac iterations
     * \param nIterations the number of iterations to perform
     *
     * The function is optimized to try a large number of iterations at once using OpenMP (if available).
     * Each thread will initialize its own random engine, which can be expensive for really small number of iterations.
     */
    inline void ransacIterations(int nIterations) {

        #if defined(_OPENMP)

        auto salt = _re();

        #pragma omp parallel
        {
            int nThreads = omp_get_num_threads();
            int threadId = omp_get_thread_num();
            std::default_random_engine thread_re; //one random engine per thread
            thread_re.seed(salt + threadId); //seed the generator in each thread with the thread id and the salt
            #pragma omp for
            for (int i = 0; i < nIterations; i++) {
                ransacIterationImpl(thread_re, threadId, nThreads);
            }
        }

        #else

        for (int i = 0; i < nIterations; i++) {
            ransacIterationImpl(_re, 0, 1);
        }

        #endif

    }

protected:

    using SamplingStrategyInternalT = std::conditional_t<std::is_same_v<SamplingStrategy,void>,std::tuple<>,SamplingStrategy>;

    template <typename RandomEngine>
    inline void ransacIterationImpl(RandomEngine & re, int threadId, int nThreads) {

        std::vector<int> selectedIdxs;
        std::vector<Measure> selectedMeasures(_minimumMeasures);

        if constexpr (std::is_same_v<SamplingStrategy,void>) {
            selectedIdxs = select_samples(re);
        } else {
            selectedIdxs = _samplingStrategy(re,_minimumMeasures,threadId,nThreads);
        }

        for (int i = 0; i < _minimumMeasures; i++) {
            selectedMeasures[i] = _measures[selectedIdxs[i]];
        }

        Model candidate(selectedMeasures);

        int nCandidateInlier = 0;
        std::vector<bool> candidateInliers(_measures.size());

        for (int i = 0; i < _measures.size(); i++) {
            Measure const& m = _measures[i]; //enfore model be const, for thread safety.
            double error = candidate.error(m);

            if (error < _threshold) {
                candidateInliers[i] = true;
                nCandidateInlier ++;
            } else {
                candidateInliers[i] = false;
            }
        }


        #if defined(_OPENMP)
        if (nCandidateInlier > _nInliers) { //most of the time the model will not be better, so check once before locking the mutex
                                            //_nInliers can onle grow, so no risk of missing something if nCandidateInlier < _nInliers
            _selected_access_locker.lock();
            if (nCandidateInlier > _nInliers) { //need to check again after locking the mutex, just to avoid race conditions
                _nInliers = nCandidateInlier;
                _currentInliers = candidateInliers;
                _currentModel = candidate;
            }
            _selected_access_locker.unlock();
        }
        #else
        if (nCandidateInlier > _nInliers) {
            _nInliers = nCandidateInlier;
            _currentInliers = candidateInliers;
            _currentModel = candidate;
        }
        #endif
    }

    template <typename RandomEngine>
    inline std::vector<int> select_samples(RandomEngine & re) const {
        size_t range = _measures.size()-1;

        std::vector<int> ret(_minimumMeasures);

        for (int i = 0; i < _minimumMeasures; i++) {
            std::uniform_int_distribution<int> dist(0,range);
            int idx = dist(re);
            for (int j = 0; j < i; j++) {
                idx += (ret[j] <= idx) ? 1 : 0;
            }
            ret[i] = idx;
            range--;
        }

        return ret;
    }

    std::vector<Measure> _measures;
    int _minimumMeasures;
    double _threshold;

    std::mutex _selected_access_locker;
    int _nInliers;
    std::vector<bool> _currentInliers;
    Model _currentModel;
    SamplingStrategyInternalT _samplingStrategy;

    std::default_random_engine _re;

};

/*!
 * \brief The FullyPrioritizedRansacSamplingStrategy class samples the measure in ransac following a strict ordering of priority
 *
 * The order of priority is given by a list of indexes
 *
 * A set of n index is selected each time operator() is called, a set of idx is built.
 * Sets of idx are built only when sets containing indices with higher priority have already been built.
 * A set has priority on another set if its lowest priority index has a higher priority than the other set.
 * In case of a tie, the priority is measured on the second lowest priority index, and so one and so forth.
 *
 * When used with multiple thread, the function becomes blocking.
 *
 * When called with a different set size than previous call, the iterator is reset and the top priority set is returned again.
 */
class FullyPrioritizedRansacSamplingStrategy {
public:
    FullyPrioritizedRansacSamplingStrategy() :
        _idxsPriorityOrder()
    {

    }
    FullyPrioritizedRansacSamplingStrategy(std::vector<int> const& idxsPriorityOrder) :
        _idxsPriorityOrder(idxsPriorityOrder)
    {

    }
    FullyPrioritizedRansacSamplingStrategy(std::vector<int> && idxsPriorityOrder) :
        _idxsPriorityOrder(idxsPriorityOrder)
    {

    }
    FullyPrioritizedRansacSamplingStrategy(FullyPrioritizedRansacSamplingStrategy const& other) :
        _idxsPriorityOrder(other._idxsPriorityOrder),
        _currentHeadsPos(other._currentHeadsPos)
    {

    }
    FullyPrioritizedRansacSamplingStrategy(FullyPrioritizedRansacSamplingStrategy && other) :
        _idxsPriorityOrder(std::move(other._idxsPriorityOrder)),
        _currentHeadsPos(std::move(other._currentHeadsPos))
    {

    }

    template<typename ReT>
    std::vector<int> operator()(ReT const& re, int nSamples, int threadId, int numThread) {
        (void) re;
        (void) threadId;
        (void) numThread;

        std::vector<int> selected(nSamples);
        int maxIdx =_idxsPriorityOrder.size()-1;

        {
            const std::lock_guard<std::mutex> lock(_locker); //function lock to garantee no two thread can get the same samples

            if (_currentHeadsPos.size() != nSamples) {
                _currentHeadsPos.resize(nSamples);
                for (int i = 0; i < nSamples; i++) {
                    _currentHeadsPos[i] = i;
                }
            } else {
                for (int i = 0; i < nSamples; i++) {
                    if (i == nSamples-1) {
                        _currentHeadsPos[i] += 1;
                        for (int j = 0; j < i; j++) { //reset idxs below to highest priority
                            _currentHeadsPos[j] = j;
                        }
                    } else if (_currentHeadsPos[i]+1 < _currentHeadsPos[i+1]) {
                        _currentHeadsPos[i] += 1;
                        for (int j = 0; j < i; j++) { //reset idxs below to highest priority
                            _currentHeadsPos[j] = j;
                        }
                        break;
                    }
                }
            }

            for (int i = 0; i < nSamples; i++) {
                selected[i] = _idxsPriorityOrder[std::min(_currentHeadsPos[i], maxIdx)];
            }
        }

        return selected;
    }

protected:
    std::mutex _locker;
    std::vector<int> _currentHeadsPos;
    std::vector<int> _idxsPriorityOrder;
};

} // namespace Optimization
} // namespace StereoVision

#endif // GENERIC_RANSAC_H
