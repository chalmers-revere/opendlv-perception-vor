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
  if ( (0 == commandlineArguments.count("cid")) || (0 == commandlineArguments.count("width")) || (0 == commandlineArguments.count("height")) || (0 == commandlineArguments.count("bpp")) || (0 == commandlineArguments.count("haarfile")) ) {
    std::cerr << argv[0] << " accesses video data using shared memory and sends it as a stream over an OD4 session." << std::endl;
    std::cerr << "         --verbose:          enable diagnostic output" << std::endl;
    std::cerr << "         --croptop:          value for cropping image from top" << std::endl;
    std::cerr << "         --croptophorizon:   value for cropping image from top" << std::endl;
    std::cerr << "         --cropbottom:       value for cropping image from bottom" << std::endl;
    std::cerr << "         --width:            the width of the image inside the shared memory" << std::endl;
    std::cerr << "         --height:           the height of the image inside the shared memory" << std::endl;
    std::cerr << "         --widthscaled:      the width of the image used in image processing" << std::endl;
    std::cerr << "         --heightscaled:     the height of the image used in image processing" << std::endl;
    std::cerr << "         --bpp:              the bits per pixel of the image inside the shared memory" << std::endl;
    std::cerr << "         --haarfile:         name of the file containing the haar classifier" << std::endl;
    std::cerr << "         --shmin:            name of the shared memory to read from (as consumer) the input image" << std::endl;
    std::cerr << "         --shmout:           name of the shared memory to write to (as producer) the output image" << std::endl;
    std::cerr << "         --widthout:         the width of the image inside the shared output memory" << std::endl;
    std::cerr << "         --heightout:        the height of the image inside the shared output memory" << std::endl;
    std::cerr << "         --disablekiwi:      do not detect Kiwi" << std::endl;
    std::cerr << "         --disablelane:      do not detect lanes" << std::endl;
    std::cerr << "         --disablepurplebox: do not detect purple boxes" << std::endl;
    std::cerr << "         --disablebluebox:   do not detect blue boxes" << std::endl;
    std::cerr << "         --linethreshold:    Hough line detector threshold (default 70)" << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=111 --width=1280 --height=960 --bpp=24 --widthscaled=256 --heightscaled=192 --widthout=128 --heightout=96 --shmin=cam0 --shmout=output --haarfile=myHaarFile.xml --croptop=10 --cropbottom=20 --croptophorizon=50 --verbose" << std::endl;
    retCode = 1;
  } else {
    bool const VERBOSE{commandlineArguments.count("verbose") != 0};

    bool const DISABLE_KIWI{commandlineArguments.count("disablekiwi") != 0};
    bool const DISABLE_LANE{commandlineArguments.count("disablelane") != 0};
    bool const DISABLE_PURPLE_BOX{commandlineArguments.count("disablepurplebox") != 0};
    bool const DISABLE_BLUE_BOX{commandlineArguments.count("disablebluebox") != 0};
    uint32_t const LINETHRESHOLD{(commandlineArguments["linethreshold"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["linethreshold"])) : 70};

    uint32_t const WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
    uint32_t const HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
    uint32_t const BPP{static_cast<uint32_t>(std::stoi(commandlineArguments["bpp"]))};
    std::string const HAARFILE{commandlineArguments["haarfile"]};

    std::string const SHMIN_NAME{(commandlineArguments["shmin"].size() > 0) ? commandlineArguments["shmin"] : "/cam0"};
    std::string const SHMOUT_NAME{(commandlineArguments["shmout"].size() > 0) ? commandlineArguments["shmout"] : "/output"};
    
    uint32_t const CROP_TOP{(commandlineArguments["croptop"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["croptop"])) : 0};
    uint32_t const CROP_BOTTOM{(commandlineArguments["cropbottom"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["cropbottom"])) : 0};
    uint32_t const CROP_TOP_HORIZON{(commandlineArguments["croptophorizon"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["croptophorizon"])) : 0};

    uint32_t const WIDTH_SCALED{(commandlineArguments["widthscaled"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["widthscaled"])) : WIDTH};
    uint32_t const HEIGHT_SCALED{(commandlineArguments["heightscaled"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["heightscaled"])) : HEIGHT};
    
    uint32_t const WIDTH_OUT{(commandlineArguments["widthout"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["widthout"])) : WIDTH};
    uint32_t const HEIGHT_OUT{(commandlineArguments["heightout"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["heightout"])) : HEIGHT};

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
      }

      uint32_t const KIWI_ID = 0;
      uint32_t const BLUE_BOX_ID = 1;
      uint32_t const PURPLE_BOX_ID = 2;
      uint32_t const LANE_MARKING_ID = 3;

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

        cv::Mat originalScaledImage;
        {
          sharedMemoryIn->lock();
          cv::Mat sourceImage = cv::cvarrToMat(image, false);
          cv::resize(sourceImage, originalScaledImage, cv::Size(WIDTH_SCALED, HEIGHT_SCALED), 0, 0, cv::INTER_NEAREST);
          sharedMemoryIn->unlock();
        }

        cv::Mat objectImage = originalScaledImage.rowRange(CROP_TOP, CROP_BOTTOM);
        cv::Mat lineImage = originalScaledImage.rowRange(CROP_TOP_HORIZON, CROP_BOTTOM);

        // Note: originalScaledImage will here after be drawn at.

        // Detect Kiwis.
        if (!DISABLE_KIWI)
        {
          auto detections = kiwiDetector.detect(objectImage);
          for (auto detection : detections) {
            cv::Rect2d d = detection;
            float angle = static_cast<float>(1.0 - (d.x + d.width / 2.0) / (WIDTH_SCALED / 2.0));
            float distance = static_cast<float>((1.0 - (d.y + d.height) / HEIGHT_SCALED_CROPPED) / 0.2);

            cv::rectangle(originalScaledImage, cv::Point(detection.tl().x, detection.tl().y + CROP_TOP), cv::Point(detection.br().x, detection.br().y + CROP_TOP), cv::Scalar(0, 0, 150), 3, 8, 0);

            opendlv::logic::sensation::Point kiwiDetection;
            kiwiDetection.azimuthAngle(angle);
            kiwiDetection.distance(distance);
            od4.send(kiwiDetection, cluon::time::now(), KIWI_ID);

            if (VERBOSE) {
              std::clog << argv[0] << ": Detected Kiwi at angle " << angle << " and distance " << distance << std::endl;
            }
          }
        }

        cv::Mat hsv;
        cv::cvtColor(objectImage, hsv, CV_BGR2HSV);

        int erosionSize = 2;
        cv::Mat element = getStructuringElement(cv::MORPH_RECT, 
            cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1), 
            cv::Point(erosionSize, erosionSize));

        // Detect blue boxes.
        if (!DISABLE_BLUE_BOX)
        {
          cv::Mat blueBoxes;
          cv::inRange(hsv, cv::Scalar(85, 140, 140), cv::Scalar(125, 255, 255), blueBoxes);

          cv::Mat blueBoxMask;
          cv::erode(blueBoxes, blueBoxMask, element);
          cv::dilate(blueBoxMask, blueBoxMask, element);

          std::vector<std::vector<cv::Point>> contoursBlue;
          std::vector<cv::Vec4i> hierarchyBlue;

          findContours(blueBoxMask, contoursBlue, hierarchyBlue, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
          
          double maxSize = 0;
          int32_t iLabel = 0;
          if (contoursBlue.size() != 0) {
            for (uint32_t i = 0; i < contoursBlue.size(); i++) {
              double area = contourArea(contoursBlue[i], false);

              if (area > maxSize) {
                maxSize = area;
                iLabel = i;
              }
            }
          
            cv::Rect r = boundingRect(contoursBlue[iLabel]);
            cv::rectangle(originalScaledImage, cv::Point(r.tl().x, r.tl().y + CROP_TOP), cv::Point(r.br().x, r.br().y + CROP_TOP), cv::Scalar(150, 0, 0), 3, 8, 0);

            float angle = static_cast<float>(1.0 - (r.x + r.width / 2.0) / (WIDTH_SCALED / 2.0));
            float distance = static_cast<float>((1.0 - (r.y + r.height) / HEIGHT_SCALED_CROPPED) / 0.2);

            opendlv::logic::sensation::Point blueDetection;
            blueDetection.azimuthAngle(angle);
            blueDetection.distance(distance);
            od4.send(blueDetection, cluon::time::now(), BLUE_BOX_ID);

            if (VERBOSE) {
              std::clog << argv[0] << ": Detected blue box at angle " << angle << " and distance " << distance << std::endl;
            }
          }
        }

        // Detect purple boxes.
        if (!DISABLE_PURPLE_BOX)
        {
          cv::Mat purpleBoxes;
          cv::inRange(hsv, cv::Scalar(115, 40, 160), cv::Scalar(155, 255, 255), purpleBoxes);

          cv::Mat purpleBoxMask;
          cv::erode(purpleBoxes, purpleBoxMask, element);
          cv::dilate(purpleBoxMask, purpleBoxMask, element);

          std::vector<std::vector<cv::Point>> contoursPurple;
          std::vector<cv::Vec4i> hierarchyPurple;
          
          findContours(purpleBoxMask, contoursPurple, hierarchyPurple, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
          
          double maxSize = 0;
          int32_t iLabel = 0;
          if (contoursPurple.size() != 0) {
            for (uint32_t i = 0; i < contoursPurple.size(); i++) {
              double area = contourArea(contoursPurple[i], false);

              if (area > maxSize) {
                maxSize = area;
                iLabel = i;
              }
            }
          
            cv::Rect r = boundingRect(contoursPurple[iLabel]);
            cv::rectangle(originalScaledImage, cv::Point(r.tl().x, r.tl().y + CROP_TOP), cv::Point(r.br().x, r.br().y + CROP_TOP), cv::Scalar(100, 0, 150), 3, 8, 0);

            float angle = static_cast<float>(1.0 - (r.x + r.width / 2.0) / (WIDTH_SCALED / 2.0));
            float distance = static_cast<float>((1.0 - (r.y + r.height) / HEIGHT_SCALED_CROPPED) / 0.2);

            opendlv::logic::sensation::Point purpleDetection;
            purpleDetection.azimuthAngle(angle);
            purpleDetection.distance(distance);
            od4.send(purpleDetection, cluon::time::now(), PURPLE_BOX_ID);

            if (VERBOSE) {
              std::clog << argv[0] << ": Detected purple box at angle " << angle << " and distance " << distance << std::endl;
            }
          }
        }

        // Detect lanes.
        if (!DISABLE_LANE)
        {
          cv::Mat hsvLines;
          cv::cvtColor(lineImage, hsvLines, CV_BGR2HSV);
          
          cv::Mat lanes;
          cv::inRange(hsvLines, cv::Scalar(20, 70, 170), cv::Scalar(60, 255, 255), lanes);

          cv::Mat edges;
          cv::Canny(lanes, edges, 50, 200, 3); 

          std::vector<cv::Vec2f> detectedLines;
          cv::HoughLines(lanes, detectedLines, 1, 5.0 * 3.14/180.0, LINETHRESHOLD);

          for (auto v : detectedLines) {
            float rho = v[0];
            float theta = v[1];
            cv::Point pt1;
            cv::Point pt2;
            double a = cos(theta);
            double b = sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a)) + CROP_TOP_HORIZON;
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a)) + CROP_TOP_HORIZON;
            cv::line(originalScaledImage, pt1, pt2, cv::Scalar(0, 150, 150), 3, CV_AA);

            opendlv::logic::sensation::Point lineDetection;
            lineDetection.azimuthAngle(theta);
            lineDetection.distance(rho);
            od4.send(lineDetection, cluon::time::now(), LANE_MARKING_ID);
              
            if (VERBOSE) {
              std::clog << argv[0] << ": Detected lane marking at polar coordinates with angle " << theta << " and distance " << rho << std::endl;
            }
          }
        }

        // Share the results.
        cv::Mat outImage;
        cv::resize(originalScaledImage, outImage, cv::Size(WIDTH_OUT, HEIGHT_OUT), 0, 0, cv::INTER_NEAREST);
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

