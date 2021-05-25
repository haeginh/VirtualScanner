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
    cout << "\t-pcd  [0/1]    : quick print of point cloud w/xphoto/inpainting.hppo color in PCD [0:object fixed/1:camera fixed]" << endl;
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
#include "vtkTransform.h"
    vtkSmartPointer<vtkCellLocator> tree = vtkSmartPointer<vtkCellLocator>::New();
    tree->SetDataSet(data);
    tree->CacheCellBoundsOn();
    tree->SetTolerance(0.0);
    tree->SetNumberOfCellsPerBucket(1);
    tree->AutomaticOn();
    tree->BuildLocator();
    tree->Update();
    cout << "done" << endl;

    srand(time(nullptr));

    // Virtual camera parameters
    double viewray0[3] = {0.0, 0.0, 1.0};
    double up0[3] = {0.0, 1.0, 0.0}; //normalize

    //Initialize LINEMOD detector
    cv::Ptr<cv::linemod::Detector> detector = readLinemod("mother.yml");
    cout << "read mother detector.." << detector->numTemplates() << " templates" << endl;
    cv::Ptr<cv::linemod::Detector> detector1 = readLinemod("daughter.yml");
    cout << "read daughter detector.." << detector1->numTemplates() << " templates" << endl;
    int num_modalities = (int)detector->getModalities().size();
    map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> labels = readLabel("mother.label");
    map<pair<string, int>, pair<vtkQuaterniond, cv::Point2i>> labels1 = readLabel("daughter.label");

    cout << "set ROI..." << flush;
    int maxX(0), maxY(0);
    for (auto label : labels)
    {
        maxX = maxX > label.second.second.x ? maxX : label.second.second.x;
        maxY = maxY > label.second.second.y ? maxY : label.second.second.y;
    }
    cv::Rect roi(cv::Point2i(scan.res_hor * 0.5 - maxX - 20, scan.res_vert * 0.5 - maxY - 20),
                 cv::Point2i(scan.res_hor * 0.5 + maxX + 20, scan.res_vert * 0.5 + maxY + 20));
    cout << "done" << endl;

    double eye[3] = {0, 0, -1};
    vector<double *> screen;
    for (int n = 0; n < scan.res_vert * scan.res_hor; n++)
        screen.push_back(new double[3]{0, 0, 0});

    function<void(double)> generateScreen = [&](double d)
    {
        eye[2] = -d;
        // Screen parameters
        double hor_len = (scan.screen_dist + d) * 2. * tan(scan.fov_hor * 0.5);
        double vert_len = (scan.screen_dist + d) * 2. * tan(scan.fov_vert * 0.5);
        double hor_interval = hor_len / (double)scan.res_hor;
        double vert_interval = vert_len / (double)scan.res_vert;
        double lu_x = -hor_len * 0.5;
        double lu_y = -vert_len * 0.5;
        double lu_z = scan.screen_dist;
        for (int vert = 0, n = 0; vert < scan.res_vert; vert++)
        {
            for (int hor = 0; hor < scan.res_hor; hor++, n++)
            {
                screen[n][0] = lu_x + hor_interval * hor;
                screen[n][1] = lu_y + vert_interval * vert;
                screen[n][2] = lu_z;
            }
        }
    };

    double viewRay[3] = {0, 0, 1};

    int failCount(0);
    vtkTransform *tr = vtkTransform::New();
    cv::Rect bBox_all(cv::Point2i(scan.res_vert * 0.5, scan.res_hor * 0.5), cv::Point2i(scan.res_vert * 0.5, scan.res_hor * 0.5));
    int maxH(scan.res_hor * 0.5), minH(scan.res_hor * 0.5), maxV(scan.res_vert * 0.5), minV(scan.res_vert * 0.5);
    cv::Point2i center(maxH, maxV);
    function<void(vtkQuaterniond, cv::Mat &, cv::Mat &)> generateImg = [&](vtkQuaterniond q, cv::Mat &color, cv::Mat &depth)
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
        ushort *depth_data = (ushort *)depth.data;

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
                    depth_data[n] = vtkMath::Dot(ep, viewRay1);
                    EstimateColor(data, cellId, x, rgb);
                    color_data[n * 3 + 0] = floor(rgb[2] + 0.5);
                    color_data[n * 3 + 1] = floor(rgb[1] + 0.5);
                    color_data[n * 3 + 2] = floor(rgb[0] + 0.5);
                }
            } // Horizontal
        }     // Vertical
    };

    Timer timer, timer1;
    int id(0);
    function<double(cv::Mat &, cv::Mat &, cv::Mat &, vtkQuaterniond)> detection = [&](cv::Mat &color, cv::Mat &depth, cv::Mat &display, vtkQuaterniond q) -> double
    {
        std::vector<cv::Mat> sources;
        sources.push_back(color);
        sources.push_back(depth);
        sources[1] = (sources[1] - minDist) * depthFactor;

        std::vector<cv::linemod::Match> matches;
        std::vector<cv::String> class_ids;
        std::vector<cv::Mat> quantized_images;
        timer.start();
        detector->match(sources, 65., matches, class_ids, quantized_images);
        timer.stop();
        color.copyTo(display);
        cv::putText(display, to_string(id++) + " (fail: " + to_string(failCount) + ")",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 255., 255.), 1);
        function<double(vtkQuaterniond, vtkQuaterniond)> getAngle = [](vtkQuaterniond a, vtkQuaterniond b) -> double
        {
            double dotProd = a.GetW() * b.GetW() + a.GetX() * b.GetX() + a.GetY() * b.GetY() + a.GetZ() * b.GetZ();
            return 2 * acos(fabs(dotProd)) * 180 / pi;
        };
        cv::linemod::Match match0;
        if (matches.size())
        {
            match0 = matches[0];
        }
        else
        {
            failCount++;
            return -1;
        }

        for (auto m : matches)
            class_ids.push_back(m.class_id);

        timer1.start();
        detector1->match(sources, 80., matches, class_ids, quantized_images);
        timer1.stop();

        pair<vtkQuaterniond, cv::Point2i> label1;
        double similarity;
        if (matches.size() && match0.similarity < matches[0].similarity)
        {
            label1 = labels1[make_pair(matches[0].class_id, matches[0].template_id)];
            cv::putText(display, "similarity2: " + to_string(matches[0].similarity) + " (" + to_string(getAngle(label1.first, q)) + " deg.)",
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
            cv::putText(display, "center diff2: " + to_string(matches[0].x + label1.second.x - center.x) + ", " + to_string(matches[0].y + label1.second.y - center.y),
                        cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
            const std::vector<cv::linemod::Template> &templates = detector1->getTemplates(matches[0].class_id, matches[0].template_id);
            drawResponse(templates, num_modalities, display, cv::Point2i(matches[0].x, matches[0].y), detector1->getT(0));
            similarity = matches[0].similarity;
        }
        else
        {
            cv::putText(display, "similarity2: -",
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
            cv::putText(display, "center diff2: -",
                        cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
            const std::vector<cv::linemod::Template> &templates = detector->getTemplates(match0.class_id, match0.template_id);
            drawResponse(templates, num_modalities, display, cv::Point2i(match0.x, match0.y), detector->getT(0));
            similarity = match0.similarity;
        }

        pair<vtkQuaterniond, cv::Point2i> label0 = labels[make_pair(match0.class_id, match0.template_id)];

        cv::putText(display, "similarity1: " + to_string(match0.similarity) + " (" + to_string(getAngle(label0.first, q)) + " deg.)",
                    cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
        cv::putText(display, "center diff1: " + to_string(match0.x + label0.second.x - center.x) + ", " + to_string(match0.y + label0.second.y - center.y),
                    cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
        cv::putText(display, "detect time: " + to_string(timer.time()) + " / " + to_string(timer1.time()),
                    cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255., 0., 0.), 1);
        return similarity;
    };

    //sample a depth
    double iso_depth = 3000;
    generateScreen(iso_depth);
    double averageS;
    int counter(0);

    while (1)
    { //sample a random quaternion
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double u3 = (double)rand() / RAND_MAX;
        double w = sqrt(1 - u1) * sin(2 * pi * u2);
        double x = sqrt(1 - u1) * cos(2 * pi * u2);
        double y = sqrt(u1) * sin(2 * pi * u3);
        double z = sqrt(u1) * cos(2 * pi * u3);
        vtkQuaterniond q = vtkQuaterniond(w, x, y, z);

        cv::Mat depth(scan.res_vert, scan.res_hor, CV_16U, cv::Scalar::all(7000));
        cv::Mat color(scan.res_vert, scan.res_hor, CV_8UC3, cv::Scalar(255, 255, 255));
        generateImg(q, color, depth);

        cv::Mat display;
        averageS += detection(color, depth, display, q);
        counter++;

        char key = cv::waitKey(1);
        if (key == 'p')
            cv::waitKey(0);
        else if(key == 'd'){
            cout<<"average smilarity for distance "<<iso_depth<<" mm: "<<averageS/counter<<" ("<<counter<<")"<<endl;
            cout<<"type new distance: "<<flush;
            cin>>iso_depth;
            generateScreen(iso_depth);
            counter = 0;
            averageS = 0;
        }
        else if (key == 'q')
            break;

        cv::imshow("display", display);
    }
    return 0;
}

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