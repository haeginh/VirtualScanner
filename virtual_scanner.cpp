/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

/**
  * \author Radu Bogdan Rusu
  *
  * @b virtual_scanner takes in a .ply or a .vtk file of an object model, and virtually scans it
  * in a raytracing fashion, saving the end results as PCD (Point Cloud Data) files. In addition,
  * it noisifies the PCD models, and downsamples them.
  * The viewpoint can be set to 1 or multiple views on a sphere.
  */
#include <string>
#include <random>
#include "boost.h"
#include "vtkQuaternion.h"
#include "vtkGeneralTransform.h"
#include "vtkTransform.h"
#include "vtkCellLocator.h"

#include "functions.cpp"

using namespace pcl;

#define EPS 0.00001

const double pi = 3.14159265358979f;
struct ScanParameters
{
    double fov_vert;    // vertical fov in radian
    double fov_hor;     // horizontal fov in radian
    int res_vert;       // vertical resolution
    int res_hor;        // horizontal  resolution
    double distance;    // distance in mm
    double screen_dist; // screen distance in mm
    int rotInterval;    // rotation interval in degree
};
using namespace std;
void PrintUsage()
{
    cout << "Usage: ./scanner [options] model.ply viewpoints.ply" << endl;
    cout << "[options]" << endl;
    cout << "\t-dist <double> : scan distance in mm (default:3000)" << endl;
    cout << "                   or put three numbers for multiple distances [start] [interval] [#]" << endl;
    cout << "\t-rot  <int>    : rotation interval in degree (default:10)" << endl;
    cout << "\t-path <string> : directory path for result files (default: .)" << endl;
    cout << "\t-cv   [depth/color/both] : print openCV matrix in YML" << endl;
    cout << "\t-ply  [0/1]    : print point cloud w/ color in PLY [0:object fixed/1:camera fixed]" << endl;
    cout << "\t-pcd  [0/1]    : quick print of point cloud w/o color in PCD [0:object fixed/1:camera fixed]" << endl;
    exit(1);
}
void PrintKeyUsage()
{
    cout << "[Key Usage]" << endl;
    cout << "\tw : write template" << endl;
    cout << "\tv : to next view" << endl;
    cout << "\tf : show the number of failed templates" << endl;
}

void EstimateColor(vtkPolyData *data, vtkIdType cellId, double *point, double *rgb);
void drawResponse(const std::vector<cv::linemod::Template> &templates,
                  int num_modalities, cv::Mat &dst, cv::Point offset, int T);
static void writeLinemod(const cv::Ptr<cv::linemod::Detector> &detector, const std::string &filename);
static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string &filename);
void writeLabel(const map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> &labels, string fileName);
map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> readLabel(string fileName);

int main(int argc, char **argv)
{
    //arguements (file names)
    string modelFileName = argv[1];

    //set scan parameters
    ScanParameters scan;
    scan.fov_hor = 90. * pi / 180.;
    scan.fov_vert = 59. * pi / 180.;
    scan.res_hor = 1280;
    scan.res_vert = 720;
    scan.distance = 3000; //mm
    scan.rotInterval = 10;
    scan.screen_dist = 6000; //mm

    //read ply file
    vtkSmartPointer<vtkPolyData> data;
    vtkPLYReader *reader = vtkPLYReader::New();
    reader->SetFileName(modelFileName.c_str());
    reader->Update();
    data = reader->GetOutput();
    // Set the distance range
    double maxExt2(0);
    for (int i = 0; i < data->GetNumberOfPoints(); i++)
    {
        double origin[3] = {0, 0, 0}, point[3];
        origin, data->GetPoint(i, point);
        double dist2 = vtkMath::Distance2BetweenPoints(origin, point);
        maxExt2 = dist2 > maxExt2 ? dist2 : maxExt2;
    }
    double maxExt = sqrt(maxExt2);
    double minDist = scan.distance - maxExt - 100 < 0 ? 0 : scan.distance - maxExt - 100;
    double maxDist = scan.distance + maxExt + 100;
    double depthFactor = 2000. / (maxDist - minDist) > 1 ? 1 : 2000. / (maxDist - minDist);
    if (maxDist < 2000)
        minDist = 0;
    cout << "min. distance: " << minDist << endl;
    cout << "max. distance: " << maxDist << endl;

    cout << "Initializing mesh..." << flush;
    // Build a spatial locator for our dataset
    vtkSmartPointer<vtkCellLocator> tree = vtkSmartPointer<vtkCellLocator>::New();
    tree->SetDataSet(data);
    tree->CacheCellBoundsOn();
    tree->SetTolerance(0.0);
    tree->SetNumberOfCellsPerBucket(1);
    tree->AutomaticOn();
    tree->BuildLocator();
    tree->Update();
    cout << "done" << endl;

    // sample quaternions
    vector<vtkQuaterniond> quaternions;
    int sampleN(0);
    srand(time(nullptr));
    cout << "Sampling mother quaternions..." << flush;
    for (int i = 0; i < sampleN; i++)
    {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double u3 = (double)rand() / RAND_MAX;
        double w = sqrt(1 - u1) * sin(2 * pi * u2);
        double x = sqrt(1 - u1) * cos(2 * pi * u2);
        double y = sqrt(u1) * sin(2 * pi * u3);
        double z = sqrt(u1) * cos(2 * pi * u3);
        vtkQuaterniond q = vtkQuaterniond(w, x, y, z);
        quaternions.push_back(q);
        quaternions.push_back(q.Conjugated());
    }
    cout << "done" << endl;

    int sampleN_large(150000);
    vector<vtkQuaterniond> quaternions_large;
    cout << "Sampling daughter quaternions..." << flush;
    for (int i = 0; i < sampleN_large; i++)
    {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double u3 = (double)rand() / RAND_MAX;
        double w = sqrt(1 - u1) * sin(2 * pi * u2);
        double x = sqrt(1 - u1) * cos(2 * pi * u2);
        double y = sqrt(u1) * sin(2 * pi * u3);
        double z = sqrt(u1) * cos(2 * pi * u3);
        vtkQuaterniond q = vtkQuaterniond(w, x, y, z);
        quaternions_large.push_back(q);
        quaternions_large.push_back(q.Conjugated());
    }
    cout << "done" << endl;

    // Virtual camera parameters
    double viewray0[3] = {0.0, 0.0, 1.0};
    double up0[3] = {0.0, 1.0, 0.0}; //normalize

    // Screen parameters
    double hor_len = scan.screen_dist * 2. * tan(scan.fov_hor * 0.5);
    double vert_len = scan.screen_dist * 2. * tan(scan.fov_vert * 0.5);
    double hor_interval = hor_len / (double)scan.res_hor;
    double vert_interval = vert_len / (double)scan.res_vert;

    //Initialize LINEMOD detector
    cv::Ptr<cv::linemod::Detector> detector = readLinemod("mother.yml");
    cout << "read mother detector.." << detector->numTemplates() << " templates" << endl;
    //detector = cv::linemod::getDefaultLINEMOD();
    cv::Ptr<cv::linemod::Detector> detector1 = readLinemod("daughter.yml");
    cout << "read daughter detector.." << detector1->numTemplates() << " templates" << endl;
    //detector1 = cv::linemod::getDefaultLINEMOD();
    int num_modalities = (int)detector->getModalities().size();

    //main loop start
    // if (path != ".")
    //     system(("mkdir " + path).c_str());
    // ofstream log(path + "/labels.txt");
    // log << "maxDist " << maxDist << endl;
    // log << "minDist " << minDist << endl;
    // log << "model " << modelFileName << endl;
    // log << "viewpoint " << viewFileName << endl
    //     << endl;

    vector<double *> screen;
    double lu_x = -hor_len * 0.5;
    double lu_y = -vert_len * 0.5;
    double lu_z = scan.screen_dist - scan.distance;
    for (int vert = 0; vert < scan.res_vert; vert++)
    {
        for (int hor = 0; hor < scan.res_hor; hor++)
        {
            screen.push_back(new double[3]{lu_x + hor_interval * hor, lu_y + vert_interval * vert, lu_z});
        }
    }
    double eye[3] = {0, 0, -scan.distance};
    double viewRay[3] = {0, 0, 1};

    int failCount(0);
    vtkTransform *tr = vtkTransform::New();
    cv::Rect bBox_all(cv::Point2i(scan.res_vert * 0.5, scan.res_hor * 0.5), cv::Point2i(scan.res_vert * 0.5, scan.res_hor * 0.5));
    int maxH(scan.res_hor * 0.5), minH(scan.res_hor * 0.5), maxV(scan.res_vert * 0.5), minV(scan.res_vert * 0.5);
    cv::Point2i center(maxH, maxV);
    function<void(vtkQuaterniond, cv::Mat &, cv::Mat &, cv::Mat &)> generateImg = [&](vtkQuaterniond q, cv::Mat &color, cv::Mat &depth, cv::Mat &mask)
    {
        double axis[3];
        double angle = q.GetRotationAngleAndAxis(axis);
        tr->Identity();
        tr->RotateWXYZ(angle * (180 / pi), axis);
        tr->Inverse();

        //Start!!
        double eye1[3], viewRay1[3];
        tr->TransformPoint(eye, eye1);
        tr->TransformPoint(viewRay, viewRay1);

        // Scanning
        vector<int> rr, gg, bb;
        double p_coords[3], x[3], t, rgb[3];
        //create ::Mat for depth & color
        uchar *color_data = color.data;
        uchar *mask_data = mask.data;
        int subId;
        for (int vert = 0, n = 0; vert < scan.res_vert; vert++)
        {
            for (int hor = 0; hor < scan.res_hor; hor++, n++)
            {
                vtkIdType cellId;
                double point[3];
                tr->TransformPoint(screen[n], point);
                if (tree->IntersectWithLine(eye1, point, 0, t, x, p_coords, subId, cellId))
                {
                    double ep[3];
                    vtkMath::Subtract(x, eye1, ep);
                    depth.at<ushort>(vert, hor) = vtkMath::Dot(ep, viewRay1);
                    mask_data[n] = 255;
                    EstimateColor(data, cellId, x, rgb);
                    color_data[n * 3 + 0] = floor(rgb[2] + 0.5);
                    color_data[n * 3 + 1] = floor(rgb[1] + 0.5);
                    color_data[n * 3 + 2] = floor(rgb[0] + 0.5);
                }
            } // Horizontal
        }     // Vertical
    };

    function<cv::Point2i(cv::Ptr<cv::linemod::Detector> &, int classID, std::vector<cv::Mat>, cv::Mat &, cv::Mat &, int &templateID)> train = [&](cv::Ptr<cv::linemod::Detector> &det, int classID, std::vector<cv::Mat> sources, cv::Mat &mask, cv::Mat &display, int &templateID) -> cv::Point2i
    {
        //addTemplate (LINEMOD)
        cv::Rect bBox;
        templateID = det->addTemplate(sources, to_string(classID), mask, &bBox);
        cv::Point2i isocenter;
        if (templateID >= 0)
        {
            const std::vector<cv::linemod::Template> &templates = det->getTemplates(to_string(classID), templateID);
            drawResponse(templates, num_modalities, display, cv::Point(bBox.x, bBox.y), det->getT(0));
            isocenter = center - bBox.tl();
            if (minV > bBox.y)
                minV = bBox.y;
            if (maxV < bBox.y + bBox.height)
                maxV = bBox.y + bBox.height;
            if (minH > bBox.x)
                minH = bBox.x;
            if (maxH < bBox.x + bBox.width)
                maxH = bBox.x + bBox.width;
            bBox_all.x = minH;
            bBox_all.y = minV;
            bBox_all.width = maxH - minH;
            bBox_all.height = maxV - minV;
            cv::rectangle(display, bBox_all, cv::Scalar(255., 255., 0., 0.));
            cv::rectangle(display, bBox, cv::Scalar(255., 0., 0., 0.));
            cv::putText(display, "isocenter: " + to_string(isocenter.x) + ", " + to_string(isocenter.y),
                        bBox_all.tl() - cv::Point2i(0, 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 0), 1);
        }
        else
            failCount++;

        return isocenter;
    };

    map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> labels = readLabel("mother.label");
    int skipCounter(0);
    Timer timer;
    for (size_t i = 0; i < quaternions.size(); i++)
    { //quaternion loop
        cout << "\rScanning mother quaternion# " << i + 1 << "/" << quaternions.size() << flush;
        cv::Mat depth(scan.res_vert, scan.res_hor, CV_16U, cv::Scalar::all(0));
        cv::Mat color(scan.res_vert, scan.res_hor, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat mask(scan.res_vert, scan.res_hor, CV_8U, cv::Scalar::all(0));
        generateImg(quaternions[i], color, depth, mask);
        std::vector<cv::Mat> sources;
        sources.push_back(color);
        sources.push_back(depth);
        cv::Mat display = color;

        std::vector<cv::linemod::Match> matches;
        std::vector<cv::String> class_ids;
        std::vector<cv::Mat> quantized_images;
        sources[1] = (sources[1] - minDist) * depthFactor;
        timer.start();
        detector->match(sources, 70., matches, class_ids, quantized_images);
        timer.stop();
        if (matches.size() > 0)
        {
            pair<vtkQuaterniond, cv::Point2i> label = labels[make_pair(matches[0].class_id, 0)];
            double dotProd = label.first.GetW() * quaternions[i].GetW() + label.first.GetX() * quaternions[i].GetX() + label.first.GetY() * quaternions[i].GetY() + label.first.GetZ() * quaternions[i].GetZ();
            double quatDiff = 1 - dotProd * dotProd;
            cv::putText(display, "quatDiff: " + to_string(quatDiff),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
            cv::putText(display, "detect time: " + to_string(timer.time()),
                        cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
            if (quatDiff > 0.2)
            {
                int template_id;
                int class_id = labels.size();
                cv::Point2i iso = train(detector, class_id, sources, mask, display, template_id);
                labels[make_pair(to_string(class_id), template_id)] = make_pair(quaternions[i], iso);
            }
            else
            {
                const std::vector<cv::linemod::Template> &templates = detector->getTemplates(matches[0].class_id, matches[0].template_id);
                drawResponse(templates, num_modalities, display, cv::Point2i(matches[0].x, matches[0].y), detector->getT(0));
                cv::imshow("color", display);
                skipCounter++;
            }
        }
        else
        {
            int template_id;
            int class_id = labels.size();
            cv::Point2i iso = train(detector, class_id, sources, mask, display, template_id);
            labels[make_pair(to_string(class_id), template_id)] = make_pair(quaternions[i], iso);
        }
        cv::putText(display, "skip #" + to_string(skipCounter),
                    cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        cv::putText(display, to_string(i) + "/" + to_string(quaternions.size()) + " (failed: " + to_string(failCount) + ")",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        cv::imshow("color", display);

        char key = (char)cv::waitKey(1);
        if (key == 'q')
            break;

        switch (key)
        {
        case 'p':
            cout << endl
                 << "p -> pause press any key " << failCount << endl;
            cv::waitKey();
            break;

        default:;
        }
    }

    // writeLinemod(detector, "mother.yml");
    // writeLabel(labels, "mother.label");
    map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> labels1  = readLabel("daughter.label");
    skipCounter = 0;
    failCount = 0;
    Timer timer1;
    for (size_t i = 0; i < quaternions_large.size(); i++)
    { //quaternion loop
        cout << "\rScanning dauther quaternion# " << i + 1 << "/" << quaternions_large.size()<< flush;
        cv::Mat depth(scan.res_vert, scan.res_hor, CV_16U, cv::Scalar::all(0));
        cv::Mat color(scan.res_vert, scan.res_hor, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat mask(scan.res_vert, scan.res_hor, CV_8U, cv::Scalar::all(0));
        generateImg(quaternions_large[i], color, depth, mask);
        std::vector<cv::Mat> sources;
        sources.push_back(color);
        sources.push_back(depth);
        sources[1] = (sources[1] - minDist) * depthFactor;

        cv::Mat display = color;
        std::vector<cv::linemod::Match> matches;
        std::vector<cv::String> class_ids;
        std::vector<cv::Mat> quantized_images;
        timer.start();
        detector->match(sources, 50., matches, class_ids, quantized_images);
        timer.stop();
        int m_id = 0;
        double dotProd;
        pair<vtkQuaterniond, cv::Point2i> label;
        for (; m_id < matches.size(); m_id++)
        {
            label = labels[make_pair(matches[m_id].class_id, matches[m_id].template_id)];
            dotProd = label.first.GetW() * quaternions_large[i].GetW() + label.first.GetX() * quaternions_large[i].GetX() + label.first.GetY() * quaternions_large[i].GetY() + label.first.GetZ() * quaternions_large[i].GetZ();
            if (fabs(dotProd) > 0.93)
                break;
        }
        if (m_id < matches.size())
        {
            class_ids = {matches[m_id].class_id};
            std::vector<cv::linemod::Match> matches1;
            cv::linemod::Match match;
            timer1.start();
            detector1->match(sources, 90, matches1, class_ids, quantized_images);
            timer1.stop();
            if (m_id == matches.size())
                failCount++;
            else if ((matches1.size() == 0) || matches1[0].similarity < 99.5)
            {
                int template_id;
                cv::Point2i iso = train(detector1, atoi(matches[m_id].class_id.c_str()), sources, mask, display, template_id);
                labels1[make_pair(matches[m_id].class_id, template_id)] = make_pair(quaternions_large[i], iso);
                match = matches[m_id];
            }
            else
            {
                const std::vector<cv::linemod::Template> &templates = detector1->getTemplates(matches1[0].class_id, matches1[0].template_id);
                drawResponse(templates, num_modalities, display, cv::Point2i(matches1[0].x, matches1[0].y), detector1->getT(0));
                label = labels1[make_pair(matches1[0].class_id, matches1[0].template_id)];
                dotProd = label.first.GetW() * quaternions_large[i].GetW() + label.first.GetX() * quaternions_large[i].GetX() + label.first.GetY() * quaternions_large[i].GetY() + label.first.GetZ() * quaternions_large[i].GetZ();
                match = matches1[0];
                skipCounter++;
            }
            double theta = acos(2 * dotProd * dotProd - 1);
            cv::putText(display, "quat diff: " + to_string(1 - dotProd * dotProd) + " (" + to_string(theta / pi * 180) + "`)",
                        cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
            cv::putText(display, "center diff: " + to_string(match.x + label.second.x - center.x) + ", " + to_string(match.y + label.second.y - center.y),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
            cv::putText(display, "similarity: " + to_string(match.similarity),
                        cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        }
        else
        {
            const std::vector<cv::linemod::Template> &templates = detector->getTemplates(matches[0].class_id, matches[0].template_id);
            drawResponse(templates, num_modalities, display, cv::Point2i(matches[0].x, matches[0].y), detector1->getT(0));
            cout<<m_id<<endl;
            cv::imshow("color", display);
            cv::waitKey(0);
            failCount++;
        }
        cv::putText(display, "detect time: " + to_string(timer.time()) + " / " + to_string(timer1.time()),
                    cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        cv::putText(display, "skip #" + to_string(skipCounter) + " / fail #" + to_string(failCount),
                    cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        cv::putText(display, to_string(i) + "/" + to_string(quaternions_large.size()) + " (picked class: " + to_string(m_id) + ")",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        cv::imshow("color", display);

        char key = (char)cv::waitKey(1);
        if (key == 'q')
            break;

        switch (key)
        {
        case 'p':
            cout << endl
                 << "p -> pause press any key " << failCount << endl;
            cv::waitKey();
            break;

        default:;
        }
    }
    writeLinemod(detector1, "daughter.yml");
    writeLabel(labels1, "daughter.label");
    return 0;
}
// }
// log.close();
// writeLinemod(detector, path + "/" + "templates.yml");
// cout << endl
//      << "failed for " << failCount << " templates" << endl;*/
//     return 0;
// }

void EstimateColor(vtkPolyData *poly, vtkIdType cellId, double *point, double *rgb)
{
    double a[3], b[3], c[3];
    poly->GetPoint(poly->GetCell(cellId)->GetPointId(0), a);
    poly->GetPoint(poly->GetCell(cellId)->GetPointId(1), b);
    poly->GetPoint(poly->GetCell(cellId)->GetPointId(2), c);
    double pa[3], pb[3], pc[3];
    vtkMath::Subtract(a, point, pa);
    vtkMath::Subtract(b, point, pb);
    vtkMath::Subtract(c, point, pc);

    double va[3], vb[3], vc[3];
    vtkMath::Cross(pc, pb, va);
    double wa = vtkMath::Norm(va);
    vtkMath::Cross(pa, pc, vb);
    double wb = vtkMath::Norm(vb);
    vtkMath::Cross(pa, pb, vc);
    double wc = vtkMath::Norm(vc);
    double sum = wa + wb + wc;
    wa /= sum;
    wb /= sum;
    wc /= sum;

    double rgb_a[3] = {poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(0), 0),
                       poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(0), 1),
                       poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(0), 2)};
    double rgb_b[3] = {poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(1), 0),
                       poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(1), 1),
                       poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(1), 2)};
    double rgb_c[3] = {poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(2), 0),
                       poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(2), 1),
                       poly->GetAttributes(0)->GetArray(0)->GetComponent(poly->GetCell(cellId)->GetPointId(2), 2)};

    vtkMath::MultiplyScalar(rgb_a, wa);
    vtkMath::MultiplyScalar(rgb_b, wb);
    vtkMath::MultiplyScalar(rgb_c, wc);
    vtkMath::Add(rgb_a, rgb_b, rgb);
    vtkMath::Add(rgb, rgb_c, rgb);
}

void drawResponse(const std::vector<cv::linemod::Template> &templates,
                  int num_modalities, cv::Mat &dst, cv::Point offset, int T)
{
    static const cv::Scalar COLORS[5] = {CV_RGB(0, 0, 255),
                                         CV_RGB(0, 255, 0),
                                         CV_RGB(255, 255, 0),
                                         CV_RGB(255, 140, 0),
                                         CV_RGB(255, 0, 0)};

    for (int m = 0; m < num_modalities; ++m)
    {
        // NOTE: Original demo recalculated max response for each feature in the TxT
        // box around it and chose the display color based on that response. Here
        // the display color just depends on the modality.
        cv::Scalar color = COLORS[m];

        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cv::linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            cv::circle(dst, pt, T / 2, color);
        }
    }
}

static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string &filename)
{
    cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

static void writeLinemod(const cv::Ptr<cv::linemod::Detector> &detector, const std::string &filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    detector->write(fs);

    std::vector<cv::String> ids = detector->classIds();
    fs << "classes"
       << "[";
    for (int i = 0; i < (int)ids.size(); ++i)
    {
        fs << "{";
        detector->writeClass(ids[i], fs);
        fs << "}"; // current class
    }
    fs << "]"; // classes
}

void writeLabel(const map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> &labels, string fileName)
{
    ofstream ofs(fileName);
    for (auto label : labels)
        ofs << label.first.first << " " << label.first.second << " "
            << label.second.first.GetW() << " " << label.second.first.GetX() << " " << label.second.first.GetY() << " " << label.second.first.GetZ() << " "
            << label.second.second.x << " " << label.second.second.y << endl;
    ofs.close();
}

map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> readLabel(string fileName)
{
    ifstream ifs(fileName);
    map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> labels;
    string classID;
    int templateID;
    double w, x, y, z, i, j;
    while (ifs >> classID >> templateID >> w >> x >> y >> z >> i >> j)
        labels[make_pair(classID, templateID)] = make_pair(vtkQuaterniond(w, x, y, z), cv::Point2i(i, j));
    ifs.close();
    return labels;
}