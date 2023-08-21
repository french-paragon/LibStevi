#include <QtTest/QtTest>

#include <MultidimArrays/MultidimArrays.h>

#include "imageProcessing/convolutions.h"
#include "imageProcessing/standardConvolutionFilters.h"

using namespace StereoVision;

using MovingAxis = ImageProcessing::Convolution::MovingWindowAxis;
using BatchedIn = ImageProcessing::Convolution::BatchedInputAxis;
using BatchedOut = ImageProcessing::Convolution::BatchedOutputAxis;
using Aggregated = ImageProcessing::Convolution::AggregateWindowsAxis;

using PaddingInfos = ImageProcessing::Convolution::PaddingInfos;
using PaddingType = ImageProcessing::Convolution::PaddingType;

class TestConvolutions: public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void initTestCase();

    void testLineConvolutionFilter();

    void testImageConvolutionFilter();

    void testGaussianConvolutionFilter();

    void testSeparatedGaussianConvolutionFilter();

private:
    std::default_random_engine re;

};

void TestConvolutions::initTestCase() {
    std::random_device rd;
    re.seed(rd());
}

void TestConvolutions::testLineConvolutionFilter() {

    constexpr int serieLength = 42;
    constexpr int batchedLength = 12;
    constexpr int filterBatchedLength = 4;

    std::uniform_real_distribution<float> dist(-1,1);
    std::uniform_int_distribution<int> intDist(-100,100);

    Multidim::Array<int, 1> serie(serieLength);

    Multidim::Array<int, 2> batchedSerie(batchedLength, serieLength);

    for (int i = 0; i < serieLength; i++) {
        serie.at(i) = intDist(re);

        for (int b = 0; b < batchedLength; b++) {

            batchedSerie.at(b, i) = intDist(re);
        }
    }

    Multidim::Array<int, 1> filter(2);
    filter.at(0) = -1;
    filter.at(1) = 1;

    Multidim::Array<int, 2> batchedFilter(filterBatchedLength, 2);

    for (int b = 0; b < filterBatchedLength; b++) {
        batchedFilter.at(b,0) = -1;
        batchedFilter.at(b,1) = 1;
    }

    Multidim::Array<int, 3> batchedAggregatedFilter(batchedLength, filterBatchedLength, 2);

    for (int b1 = 0; b1 < batchedLength; b1++) {
        for (int b2 = 0; b2 < filterBatchedLength; b2++) {
            batchedAggregatedFilter.at(b1,b2,0) = -1;
            batchedAggregatedFilter.at(b1,b2,1) = 1;
        }
    }

    ImageProcessing::Convolution::Filter<int, MovingAxis> singleFilter(filter, MovingAxis(PaddingInfos()));
    Multidim::Array<int, 1> convolved = singleFilter.convolve(serie);

    QCOMPARE(convolved.shape()[0], serieLength-1);

    for (int i = 1; i < serieLength; i++) {
        int observed = convolved.value(i-1);
        int expected = serie.value(i)-serie.value(i-1);

        QCOMPARE(observed, expected);
    }

    ImageProcessing::Convolution::Filter<int, MovingAxis> singleFilterPadded(filter, MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    convolved = singleFilterPadded.convolve(serie);

    QCOMPARE(convolved.shape()[0], serieLength);

    for (int i = 1; i < serieLength; i++) {
        int observed = convolved.value(i);
        int expected = serie.value(i)-serie.value(i-1);

        QCOMPARE(observed, expected);
    }

    ImageProcessing::Convolution::Filter<int, BatchedIn, MovingAxis>
            singleFilterBatchedIn(filter, BatchedIn(), MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    Multidim::Array<int, 2> batchedConvolved = singleFilterBatchedIn.convolve(batchedSerie);

    QCOMPARE(batchedConvolved.shape()[0], batchedSerie.shape()[0]);
    QCOMPARE(batchedConvolved.shape()[1], serieLength);

    for (int i = 1; i < serieLength; i++) {

        for (int b = 0; b < batchedLength; b++) {
            int observed = batchedConvolved.value(b, i);
            int expected = batchedSerie.value(b, i)-batchedSerie.value(b, i-1);

            QCOMPARE(observed, expected);
        }
    }

    ImageProcessing::Convolution::Filter<int, BatchedOut, MovingAxis>
            singleFilterBatchedOut(batchedFilter, BatchedOut(), MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    batchedConvolved = singleFilterBatchedOut.convolve(serie);

    QCOMPARE(batchedConvolved.shape()[0], batchedFilter.shape()[0]);
    QCOMPARE(batchedConvolved.shape()[1], serieLength);

    for (int i = 1; i < serieLength; i++) {

        for (int b = 0; b < filterBatchedLength; b++) {
            int observed = batchedConvolved.value(b, i);
            int expected = serie.value(i)-serie.value(i-1);

            QCOMPARE(observed, expected);
        }
    }

    ImageProcessing::Convolution::Filter<int, BatchedIn, BatchedOut, MovingAxis>
            singleFilterBatchedInNOut(batchedFilter, BatchedIn(), BatchedOut(), MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    Multidim::Array<int, 3> doubleBatchedConvolved = singleFilterBatchedInNOut.convolve(batchedSerie);

    QCOMPARE(doubleBatchedConvolved.shape()[0], batchedSerie.shape()[0]);
    QCOMPARE(doubleBatchedConvolved.shape()[1], batchedFilter.shape()[0]);
    QCOMPARE(doubleBatchedConvolved.shape()[2], serieLength);

    for (int i = 1; i < serieLength; i++) {

        for (int b1 = 0; b1 < batchedLength; b1++) {
            for (int b2 = 0; b2 < filterBatchedLength; b2++) {
                int observed = doubleBatchedConvolved.value(b1, b2, i);
                int expected = batchedSerie.value(b1, i)-batchedSerie.value(b1, i-1);

                QCOMPARE(observed, expected);
            }
        }
    }

    ImageProcessing::Convolution::Filter<int, Aggregated, BatchedOut, MovingAxis>
            singleFilterBatchedAggregated(batchedAggregatedFilter, Aggregated(), BatchedOut(), MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    batchedConvolved = singleFilterBatchedAggregated.convolve(batchedSerie);

    QCOMPARE(batchedConvolved.shape()[0], batchedFilter.shape()[0]);
    QCOMPARE(batchedConvolved.shape()[1], serieLength);

    for (int i = 1; i < serieLength; i++) {

        for (int b = 0; b < filterBatchedLength; b++) {
            int observed = batchedConvolved.value(b, i);

            int expected = 0;
            for (int bs = 0; bs < batchedLength; bs++) {
                expected += batchedSerie.value(bs, i)-batchedSerie.value(bs, i-1);
            }

            QCOMPARE(observed, expected);
        }
    }

}

void TestConvolutions::testImageConvolutionFilter() {

    constexpr int imageHeight = 12;
    constexpr int imageWidth = 17;
    constexpr int batchedLength = 6;
    constexpr int filterBatchedLength = 3;

    std::uniform_int_distribution<int> intDist(-100,100);

    Multidim::Array<int, 2> img(imageHeight, imageWidth);

    Multidim::Array<int, 3> batchedImg(batchedLength, imageHeight, imageWidth);

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageHeight; j++) {

            img.at(i, j) = intDist(re);

            for (int b = 0; b < batchedLength; b++) {

                batchedImg.at(b, i, j) = intDist(re);
            }
        }
    }

    Multidim::Array<int, 2> filter(2, 2);
    filter.at(0,0) = 1;
    filter.at(1,1) = 1;
    filter.at(0,1) = -1;
    filter.at(1,0) = -1;

    Multidim::Array<int, 3> batchedFilter(filterBatchedLength, 2, 2);

    for (int b = 0; b < filterBatchedLength; b++) {
        batchedFilter.at(b,0,0) = 1;
        batchedFilter.at(b,1,1) = 1;
        batchedFilter.at(b,0,1) = -1;
        batchedFilter.at(b,1,0) = -1;
    }

    ImageProcessing::Convolution::Filter<int, MovingAxis, MovingAxis> singleFilter(filter, MovingAxis(PaddingInfos()), MovingAxis(PaddingInfos()));
    Multidim::Array<int, 2> convolved = singleFilter.convolve(img);

    QCOMPARE(convolved.shape()[0], img.shape()[0]-1);
    QCOMPARE(convolved.shape()[1], img.shape()[1]-1);


    for (int i = 1; i < imageHeight; i++) {
        for (int j = 1; j < imageHeight; j++) {

            int observed = convolved.value(i-1, j-1);
            int expected = img.value(i, j)+img.value(i-1, j-1)
                    -img.value(i-1, j)-img.value(i, j-1);

            QCOMPARE(observed, expected);
        }
    }

    ImageProcessing::Convolution::Filter<int, MovingAxis, MovingAxis> singleFilterPadded(filter,
                                                                                         MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)),
                                                                                         MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    convolved = singleFilterPadded.convolve(img);

    QCOMPARE(convolved.shape()[0], img.shape()[0]);
    QCOMPARE(convolved.shape()[1], img.shape()[1]);

    for (int i = 1; i < imageHeight; i++) {
        for (int j = 1; j < imageHeight; j++) {

            int observed = convolved.value(i, j);
            int expected = img.value(i, j)+img.value(i-1, j-1)
                    -img.value(i-1, j)-img.value(i, j-1);

            QCOMPARE(observed, expected);
        }
    }

    ImageProcessing::Convolution::Filter<int, BatchedIn, MovingAxis, MovingAxis> singleFilterPaddedBatchedIn(filter,
                                                                                                             BatchedIn(),
                                                                                                             MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)),
                                                                                                             MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    Multidim::Array<int, 3> batchedConvolved = singleFilterPaddedBatchedIn.convolve(batchedImg);

    QCOMPARE(batchedConvolved.shape()[0], batchedImg.shape()[0]);
    QCOMPARE(batchedConvolved.shape()[1], batchedImg.shape()[1]);
    QCOMPARE(batchedConvolved.shape()[2], batchedImg.shape()[2]);

    for (int i = 1; i < imageHeight; i++) {
        for (int j = 1; j < imageHeight; j++) {

            for (int b = 0; b < batchedLength; b++) {
                int observed = batchedConvolved.value(b, i, j);
                int expected = batchedImg.value(b, i, j)+batchedImg.value(b, i-1, j-1)
                        -batchedImg.value(b, i-1, j)-batchedImg.value(b, i, j-1);

                QCOMPARE(observed, expected);
            }
        }
    }

    ImageProcessing::Convolution::Filter<int, BatchedOut, MovingAxis, MovingAxis> singleFilterPaddedBatchedOut(batchedFilter,
                                                                                                               BatchedOut(),
                                                                                                               MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)),
                                                                                                               MovingAxis(PaddingInfos(1,0,PaddingType::Mirror)));
    batchedConvolved = singleFilterPaddedBatchedOut.convolve(img);

    for (int i = 1; i < imageHeight; i++) {
        for (int j = 1; j < imageHeight; j++) {

            for (int b = 0; b < filterBatchedLength; b++) {
                int observed = batchedConvolved.value(b, i, j);
                int expected = img.value(i, j)+img.value(i-1, j-1)
                        -img.value(i-1, j)-img.value(i, j-1);

                QCOMPARE(observed, expected);
            }
        }
    }
}

void TestConvolutions::testGaussianConvolutionFilter() {

    constexpr float sigma = 1;
    constexpr int radius = 3;
    constexpr bool normalize = true;

    constexpr int size = 2*radius+1;

    constexpr int imageHeight = size;
    constexpr int imageWidth = 2*size+1;
    constexpr int imageChannels = 3;

    Multidim::Array<float, 3> img(imageHeight, imageWidth, imageChannels);

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {

            float mult = 1;

            if (j == size) {
                mult = 0;
            }

            if (j > size) {
                mult = -1;
            }

            for (int c = 0; c < imageChannels; c++) {

                img.at(i, j, c) = mult*(c+1);
            }
        }
    }

    ImageProcessing::Convolution::Filter<float, MovingAxis, MovingAxis, BatchedIn> gaussianFilter =
            ImageProcessing::Convolution::uniformGaussianFilter(sigma, radius, normalize, MovingAxis(PaddingInfos()), MovingAxis(PaddingInfos()), BatchedIn());

    Multidim::Array<float, 3> filtered = gaussianFilter.convolve(img);

    QCOMPARE(filtered.shape()[0], 1);
    QCOMPARE(filtered.shape()[1], size+2);
    QCOMPARE(filtered.shape()[2], imageChannels);


    for (int c = 0; c < imageChannels; c++) {

        float val = c+1;

        QCOMPARE(filtered.valueUnchecked(0,0, c), val);
        QCOMPARE(filtered.valueUnchecked(0,radius+1, c), static_cast<float>(0));
        QCOMPARE(filtered.valueUnchecked(0,size+1, c), -val);
    }


    Multidim::Array<float, 2> dirac(size, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float val = 0;

            if (i == radius and j == radius) {
                val = 1;
            }

            dirac.atUnchecked(i,j) = val;
        }
    }


    ImageProcessing::Convolution::Filter<float, MovingAxis, MovingAxis> gaussianFlatFilter =
            ImageProcessing::Convolution::uniformGaussianFilter(sigma, radius, false,
                                                                MovingAxis(PaddingInfos(radius, radius)),
                                                                MovingAxis(PaddingInfos(radius, radius)));

    gaussianFlatFilter.setPaddingConstant(0);

    Multidim::Array<float, 2> response = gaussianFlatFilter.convolve(dirac);

    QCOMPARE(response.shape()[0], size);
    QCOMPARE(response.shape()[1], size);

    float var = sigma*sigma;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float val = response.valueUnchecked(i,j);

            int di = i-radius;
            int dj = j-radius;

            float expected = std::exp(-(di*di)/var)*std::exp(-(dj*dj)/var);

            float tol = 1e-4;
            float delta = std::abs(val - expected);

            QVERIFY2(delta < tol, qPrintable(QString("Different filters results, error = %1.").arg(delta)));
        }
    }

}

void TestConvolutions::testSeparatedGaussianConvolutionFilter() {

    constexpr float sigma = 1;
    constexpr int radius = 3;
    constexpr bool normalize = true;

    constexpr int size = 2*radius+1;

    constexpr int imageHeight = size;
    constexpr int imageWidth = 2*size+1;
    constexpr int imageChannels = 3;

    std::uniform_real_distribution<float> dist(-1,1);

    Multidim::Array<float, 3> img(imageHeight, imageWidth, imageChannels);

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {

            for (int c = 0; c < imageChannels; c++) {

                img.at(i, j, c) = dist(re);
            }
        }
    }

    using FiltType = ImageProcessing::Convolution::Filter<float, MovingAxis, MovingAxis, BatchedIn>;

    PaddingInfos pad1(radius-1, ImageProcessing::Convolution::PaddingType::Mirror);
    PaddingInfos pad2(radius-2, ImageProcessing::Convolution::PaddingType::Periodic);

    FiltType gaussianFilter =
            ImageProcessing::Convolution::uniformGaussianFilter(sigma, radius, normalize, MovingAxis(pad1), MovingAxis(pad2), BatchedIn());

    std::array<FiltType, FiltType::nAxesOfType(ImageProcessing::Convolution::AxisType::Moving)> separatedGaussianFilter =
            ImageProcessing::Convolution::separatedGaussianFilters(sigma, radius, normalize, MovingAxis(pad1), MovingAxis(pad2), BatchedIn());

    Multidim::Array<float, 3> filtered = gaussianFilter.convolve(img);

    Multidim::Array<float, 3> sep_filtered = separatedGaussianFilter[0].convolve(img);

    for (int i = 1; i < separatedGaussianFilter.size(); i++) {
        sep_filtered = separatedGaussianFilter[i].convolve(sep_filtered);
    }

    QCOMPARE(filtered.shape()[0], sep_filtered.shape()[0]);
    QCOMPARE(filtered.shape()[1], sep_filtered.shape()[1]);
    QCOMPARE(filtered.shape()[2], sep_filtered.shape()[2]);

    QCOMPARE(filtered.shape()[2], imageChannels);

    for (int i = 0; i < filtered.shape()[0]; i++) {
        for (int j = 0; j < filtered.shape()[1]; j++) {

            for (int c = 0; c < filtered.shape()[2]; c++) {

                float tol = 1e-4;
                float delta = std::abs(filtered.value(i,j,c) - sep_filtered.value(i,j,c));

                QVERIFY2(delta < tol, qPrintable(QString("Different filters results, error = %1.").arg(delta)));
            }
        }
    }

}

QTEST_MAIN(TestConvolutions)
#include "testConvolution.moc"
