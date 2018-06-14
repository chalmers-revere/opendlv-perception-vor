/*
 * Copyright (C) 2018 Ola Benderius, Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include "detect-kiwi.hpp"

int32_t main(int32_t argc, char **argv) {
  int32_t retCode{0};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if ( (0 == commandlineArguments.count("cid")) || (0 == commandlineArguments.count("width")) || (0 == commandlineArguments.count("height")) || (0 == commandlineArguments.count("bpp")) || (0 == commandlineArguments.count("haarfile")) || (0 == commandlineArguments.count("shmin")) || (0 == commandlineArguments.count("shmout")) || (0 == commandlineArguments.count("hranges")) ) {
    std::cerr << argv[0] << " accesses video data using shared memory and sends it as a stream over an OD4 session." << std::endl;
    std::cerr << "         --verbose:    enable diagnostic output" << std::endl;
    std::cerr << "         --croptop:    value for cropping image from top" << std::endl;
    std::cerr << "         --cropbottom: value for cropping image from bottom" << std::endl;
    std::cerr << "         --width:      the width of the image inside the shared memory" << std::endl;
    std::cerr << "         --height:     the height of the image inside the shared memory" << std::endl;
    std::cerr << "         --widthscaled: the width of the image used in image processing" << std::endl;
    std::cerr << "         --heightscaled: the height of the image used in image processing" << std::endl;
    std::cerr << "         --bpp:        the bits per pixel of the image inside the shared memory" << std::endl;
    std::cerr << "         --haarfile:   name of the file containing the haar classifier" << std::endl;
    std::cerr << "         --shmin:      name of the shared memory to read from (as consumer) the input image" << std::endl;
    std::cerr << "         --shmout:     name of the shared memory to write to (as producer) the output image" << std::endl;
    std::cerr << "         --widthout:   the width of the image inside the shared output memory" << std::endl;
    std::cerr << "         --heightout:  the height of the image inside the shared output memory" << std::endl;
    std::cerr << "         --hranges:    list of H-color ranges used for the image processing: min0:max0,min1:max1,min2:max2,..." << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=111 --width=1280 --height=960 --bpp=24 --widthout=128 --heightout=96 --shmin=cam0 --shmout=output --hranges=1:2,3:4 --haarfile=myHaarFile.xml --croptop=10 --cropbottom=20 --verbose" << std::endl;
    retCode = 1;
  } else {
    bool const VERBOSE{commandlineArguments.count("verbose") != 0};

    uint32_t const WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
    uint32_t const HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
    uint32_t const BPP{static_cast<uint32_t>(std::stoi(commandlineArguments["bpp"]))};
    uint32_t const CROP_TOP{static_cast<uint32_t>(std::stoi(commandlineArguments["croptop"]))};
    uint32_t const CROP_BOTTOM{static_cast<uint32_t>(std::stoi(commandlineArguments["cropbottom"]))};
    std::string const HAARFILE{commandlineArguments["haarfile"]};
    std::string const SHMIN_NAME{(commandlineArguments["shmin"].size() > 0) ? commandlineArguments["shmin"] : "/cam0"};
    std::string const SHMOUT_NAME{(commandlineArguments["shmout"].size() > 0) ? commandlineArguments["shmout"] : "/output"};

    uint32_t const WIDTH_SCALED{(commandlineArguments["widthscaled"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["widthscaled"])) : 128};
    uint32_t const HEIGHT_SCALED{(commandlineArguments["heightscaled"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["heightscaled"])) : 96};
    
    uint32_t const WIDTH_OUT{(commandlineArguments["widthout"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["widthout"])) : 640};
    uint32_t const HEIGHT_OUT{(commandlineArguments["heightout"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["heightout"])) : 480};


    std::vector<std::pair<uint32_t, uint32_t> > hranges;
    for (auto a : stringtoolbox::split(commandlineArguments["hranges"], ',')) {
        auto b = stringtoolbox::split(a, ':');
        if (2 == b.size()) {
            hranges.push_back(std::make_pair(std::stoi(b.at(0)), std::stoi(b.at(1))));
        }
    }

    std::unique_ptr<cluon::SharedMemory> sharedMemoryIn(new cluon::SharedMemory{SHMIN_NAME});
    if (sharedMemoryIn && sharedMemoryIn->valid()) {
      std::clog << argv[0] << ": Found shared memory for input image '" << sharedMemoryIn->name() << "' (" << sharedMemoryIn->size() << " bytes)." << std::endl;
    } else {
      std::cerr << argv[0] << ": Failed to access shared memory '" << SHMIN_NAME << "'." << std::endl;
    }

    std::unique_ptr<cluon::SharedMemory> sharedMemoryOut(new cluon::SharedMemory{SHMOUT_NAME, WIDTH*HEIGHT*BPP/8});
    if (sharedMemoryOut && sharedMemoryOut->valid()) {
      std::clog << argv[0] << ": Created shared memory for output image '" << sharedMemoryOut->name() << "' (" << sharedMemoryOut->size() << " bytes)." << std::endl;
    } else {
      std::cerr << argv[0] << ": Failed to create shared memory '" << SHMOUT_NAME << "'." << std::endl;
    }

    if ( (sharedMemoryIn && sharedMemoryIn->valid()) && (sharedMemoryOut && sharedMemoryOut->valid()) ) {
      if (VERBOSE) {
        std::clog << argv[0] << ": Starting image processing." << std::endl;
        std::clog << argv[0] << ": Cropping " << CROP_TOP << ", " << CROP_BOTTOM << std::endl;
        for (auto b : hranges) {
        	std::clog << "  " << b.first << ".." << b.second << std::endl;
        }
      }

      DetectKiwi kiwiDetector(HAARFILE);

      cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

      uint32_t const HEIGHT_SCALED_CROPPED = HEIGHT_SCALED - CROP_TOP - CROP_BOTTOM;

      CvSize size;
      size.width = WIDTH;
      size.height = HEIGHT;

      IplImage *image = cvCreateImageHeader(size, IPL_DEPTH_8U, BPP/8);
      sharedMemoryIn->lock();
      image->imageData = sharedMemoryIn->data();
      image->imageDataOrigin = image->imageData;
      sharedMemoryIn->unlock();

      while (od4.isRunning()) {
        sharedMemoryIn->wait();

        cv::Mat processImage;
        {
          sharedMemoryIn->lock();
          cv::Mat sourceImage = cv::cvarrToMat(image, false);
          cv::resize(sourceImage, processImage, cv::Size(WIDTH_SCALED, HEIGHT_SCALED), 0, 0, cv::INTER_NEAREST);
          sharedMemoryIn->unlock();
        }

        processImage = processImage.rowRange(CROP_TOP, CROP_BOTTOM);

        // Detect Kiwis.
        auto detections = kiwiDetector.detect(processImage);
        for (auto detection : detections) {
          cv::Rect2d d = detection;
          float angle = static_cast<float>(1.0 - (d.x + d.width / 2.0) / (WIDTH_SCALED / 2.0));
          float distance = static_cast<float>((1.0 - (d.y + d.height) / HEIGHT_SCALED_CROPPED) / 0.2);

          opendlv::logic::sensation::Point kiwiDetection;
          kiwiDetection.azimuthAngle(angle);
          kiwiDetection.distance(distance);

          od4.send(kiwiDetection, cluon::time::now(), 0);
        }

        cv::Mat hsv;
        cv::cvtColor(processImage, hsv, CV_BGR2HSV);

        // Detect lanes.
        cv::Mat lanes;
        cv::inRange(hsv, cv::Scalar(20, 70, 170), cv::Scalar(60, 255, 255), lanes);

        cv::Mat edges;
        cv::Canny(lanes, edges, 50, 200, 3); 

        std::vector<cv::Vec2f> detectedLines;
        cv::HoughLines(lanes, detectedLines, 1, 5.0 * 3.14/180.0, 50.0);

        (void) detectedLines;
        // .. send in some format.


        // Share the results.
        cv::Mat outImage;
        cv::resize(processImage, outImage, cv::Size(WIDTH_OUT, HEIGHT_OUT), 0, 0, cv::INTER_NEAREST);
        {
          sharedMemoryOut->lock();
          ::memcpy(sharedMemoryOut->data(), reinterpret_cast<char*>(outImage.data), outImage.step * outImage.rows);
          sharedMemoryOut->unlock();
          sharedMemoryOut->notifyAll();
        }
      }
      cvReleaseImageHeader(&image);
    }
  }
  return retCode;
}

