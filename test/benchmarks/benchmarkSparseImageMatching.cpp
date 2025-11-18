#include <QtTest/QtTest>
#include <QDebug>

#include "correlation/matching_costs.h"
#include "imageProcessing/convolutions.h"

#include "sparseMatching/pointsDescriptors.h"

class BenchmarkSparseImageMatching: public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void initTestCase();

    void benchmarkCircularFourrierTransformDescriptor();

private:

    inline Multidim::Array<float,2> buildRandomPatch(int size, float autoCorrelationRadius = 6) {

        Multidim::Array<float,2> base(size, size);

        int kernelRadius = 2*std::ceil(autoCorrelationRadius);
        int kernelSize = 2*kernelRadius+1;

        Multidim::Array<float,2> kernel(kernelSize,kernelSize);

        for (int i = 0; i < kernelSize; i++) {

            float d1 = i - kernelRadius;
            d1 /= autoCorrelationRadius;

            for (int j = 0; j < kernelSize; j++) {

                float d2 = j - kernelRadius;
                d2 /= autoCorrelationRadius;

                kernel.atUnchecked(i,j) = std::exp(-(d1*d1 + d2*d2));
            }
        }

        std::uniform_real_distribution<float> dist(0,1);

        int dx = std::max(kernelRadius/2,1);
        std::uniform_int_distribution<int> distDxDelta(-dx,dx);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {

                base.atUnchecked(i,j) = 0;
            }
        }

        for (int i = 0; i < size; i += dx) {
            for (int j = 0; j < size; j += dx) {
                int di = distDxDelta(re);
                int dj = distDxDelta(re);

                int idxI = std::max(0,std::min(i+di,size-1));
                int idxJ = std::max(0,std::min(j+dj,size-1));

                base.atUnchecked(idxI,idxJ) = dist(re);
            }
        }

        using MovingAxis = StereoVision::ImageProcessing::Convolution::MovingWindowAxis;
        using Filter = StereoVision::ImageProcessing::Convolution::Filter<float, MovingAxis, MovingAxis>;

        StereoVision::ImageProcessing::Convolution::PaddingInfos pInfos(kernelRadius);
        MovingAxis mAxisDef(0,pInfos);
        Filter conv(kernel, mAxisDef, mAxisDef);

        return conv.convolve(base);

    }

    inline static Multidim::Array<float,2> rotatePatch(Multidim::Array<float,2> const& other) {
        Multidim::Array<float,2> ret(other.shape());

        if (other.shape()[0] != other.shape()[1]) {
            return ret;
        }

        for (int i = 0; i < other.shape()[0]; i++) {
            for (int j = 0; j < other.shape()[1]; j++) {
                ret.atUnchecked(other.shape()[0]-j-1,i) = other.valueUnchecked(i,j);
            }
        }

        return ret;
    }

    std::default_random_engine re;

};

void BenchmarkSparseImageMatching::initTestCase() {
    std::random_device rd;
    re.seed(rd());
}

void BenchmarkSparseImageMatching::benchmarkCircularFourrierTransformDescriptor() {

    using FFTDescriptor = StereoVision::SparseMatching::CircularFFTFeatureInfos<32,16,8>;

    int nPatches = 50;
    constexpr int patchSize = 21;
    constexpr float autoCorrelationRadius = 6;

    std::vector<Multidim::Array<float,2>> patches(nPatches);
    std::vector<Multidim::Array<float,2>> patchesRotated(nPatches);
    std::vector<Multidim::Array<float,2>> patchesNegative(nPatches);

    for (int i = 0; i < nPatches; i++) {
        patches[i] = buildRandomPatch(patchSize, autoCorrelationRadius);
        patchesRotated[i] = rotatePatch(patches[i]);
        patchesNegative[i] = buildRandomPatch(patchSize, autoCorrelationRadius);

        QCOMPARE(patches[i].shape()[0], patchSize);
        QCOMPARE(patches[i].shape()[1], patchSize);

        QCOMPARE(patchesRotated[i].shape()[0], patchSize);
        QCOMPARE(patchesRotated[i].shape()[1], patchSize);

        QCOMPARE(patchesNegative[i].shape()[0], patchSize);
        QCOMPARE(patchesNegative[i].shape()[1], patchSize);
    }

    std::vector<StereoVision::SparseMatching::orientedCoordinate<2>> pos(1);
    pos[0] = StereoVision::SparseMatching::orientedCoordinate<2>(std::array<int,2>{patchSize/2,patchSize/2},
                                                                 std::array<float,2>{0,0});

    QCOMPARE(pos.size(), 1);

    std::vector<float> circleRadiuses{8,4,2};

    int nPositive = 0;

    QBENCHMARK {
        nPositive = 0;
        for (int i = 0; i < nPatches; i++) {
            auto descriptor = StereoVision::SparseMatching::CircularFFTAmplitudeDescriptors<FFTDescriptor>(pos, patches[i],circleRadiuses);
            auto descriptorRotated = StereoVision::SparseMatching::CircularFFTAmplitudeDescriptors<FFTDescriptor>(pos, patchesRotated[i],circleRadiuses);
            auto descriptorNegative = StereoVision::SparseMatching::CircularFFTAmplitudeDescriptors<FFTDescriptor>(pos, patchesNegative[i],circleRadiuses);

            QCOMPARE(descriptor.size(), pos.size());
            QCOMPARE(descriptor.size(), pos.size());
            QCOMPARE(descriptor.size(), pos.size());

            auto distanceRotated = StereoVision::Correlation::SumAbsDiff(descriptor[0].features, descriptorRotated[0].features);
            auto distanceNegative = StereoVision::Correlation::SumAbsDiff(descriptor[0].features, descriptorNegative[0].features);

            if (distanceRotated < distanceNegative) {
                nPositive++;
            }
        }
    }

    QString percentPositive = QString("%1%").arg(float(nPositive)/nPatches * 100,0,'f',2);

    qInfo() << ("Percent of positive samples = " + percentPositive);
}

QTEST_MAIN(BenchmarkSparseImageMatching)
#include "benchmarkSparseImageMatching.moc"
