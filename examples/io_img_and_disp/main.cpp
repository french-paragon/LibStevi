#include <QDir>
#include <QFileInfo>
#include <QDirIterator>

#include <QTextStream>

#include "io/image_io.h"

int main() {

	QString source_dir = "@CMAKE_SOURCE_DIR@";
	QString exec_dir = "@CMAKE_CURRENT_BINARY_DIR@";

	QTextStream out(stdout);

	QDir test_data_dir(source_dir);
	QDir out_data_dir(exec_dir);

	if (!test_data_dir.exists()) {
		out << "Unable to find source folder ! Abort example running" << Qt::endl;
		return -1;
	}

	bool found = test_data_dir.cd("test/test_data/stereo_images");

	if (!found) {
		out << "Unable to find test data folder ! Abort example running" << Qt::endl;
		return -1;
	}

	QDirIterator it(test_data_dir);

	while (it.hasNext()) {
		QString path = it.next();

		out << "Processing file: " << path << Qt::endl;
		QFileInfo info(path);

		if (path.toLower().endsWith(".bmp") or
				path.toLower().endsWith(".jpg") or
				path.toLower().endsWith(".jpeg") or
				path.toLower().endsWith(".png")) {

			Multidim::Array<uint8_t, 3> img = StereoVision::IO::readImage<uint8_t>(path.toStdString());

			bool ok = StereoVision::IO::writeImage<uint8_t>(out_data_dir.filePath(info.fileName()).toStdString(), img);

			if (ok) {
				out << "\t" << "File succesfully written to disk" << Qt::endl;
			} else {
				out << "\t" << "Failed to write file to disk" << Qt::endl;
			}

		} else if (path.toLower().endsWith(".pfm")) {

			Multidim::Array<float, 3> img = StereoVision::IO::readImage<float>(path.toStdString());

			float max = 0;

			for (int i = 0; i < img.shape()[0]; i++) {
				for (int j = 0; j < img.shape()[1]; j++) {
					for (int c = 0; c < img.shape()[2]; c++) {

						if (img.atUnchecked(i,j,c) > max) {
							max = img.atUnchecked(i,j,c);
						}
					}
				}
			}

			for (int i = 0; i < img.shape()[0]; i++) {
				for (int j = 0; j < img.shape()[1]; j++) {
					for (int c = 0; c < img.shape()[2]; c++) {

						img.atUnchecked(i,j,c) *= 256./max;
					}
				}
			}

			bool ok = StereoVision::IO::writeImage<uint8_t>((out_data_dir.filePath(info.baseName()) + ".bmp" ).toStdString(), img);

			if (ok) {
				out << "\t" << "File succesfully written to disk" << Qt::endl;
			} else {
				out << "\t" << "Failed to write file to disk" << Qt::endl;
			}
		}
	}

	out << "Finished processing files" << Qt::endl;

	return 0;
}
