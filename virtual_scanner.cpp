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
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <opencv2/surface_matching.hpp>
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
cv::Mat ReadPLY(string name);

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
    // Virtual camera parameters
    double viewray0[3] = {0.0, 0.0, 1.0};
    double up0[3] = {0.0, 1.0, 0.0}; //normalize

    //read ply file
    //cv::Mat pc = cv::ppf_match_3d::loadPLYSimple(modelFileName.c_str(), 0);
    //cv::Mat pc1;
    cv::Mat pc = ReadPLY(modelFileName);

    //cv::ppf_match_3d::computeNormalsPC3d(pc, pc1, 6, false, cv::Vec3f(0,0,-scan.distance));
    //cv::ppf_match_3d::writePLY(pc, "test.ply");
    //cv::hconcat(pc, pc_normal, pc);
    Timer timer, timer1;
    cv::ppf_match_3d::PPF3DDetector detectorSurf(0.05, 0.05);
    //train the model
    cout << "Training..." << flush;
    timer.start();
    detectorSurf.trainModel(pc);
    timer.stop();
    cout << " -> " << timer.time() << "s" << endl;
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

    srand(time(nullptr));

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
    vtkTransform *trInv = vtkTransform::New();
    cv::Rect bBox_all(cv::Point2i(scan.res_vert * 0.5, scan.res_hor * 0.5), cv::Point2i(scan.res_vert * 0.5, scan.res_hor * 0.5));
    int maxH(scan.res_hor * 0.5), minH(scan.res_hor * 0.5), maxV(scan.res_vert * 0.5), minV(scan.res_vert * 0.5);
    cv::Point2i center(maxH, maxV);
    function<void(vtkQuaterniond, cv::Mat &)> generatePC = [&](vtkQuaterniond q, cv::Mat &_pc)
    {
        double axis[3];
        double angle = q.GetRotationAngleAndAxis(axis);
        tr->Identity();
        tr->RotateWXYZ(angle * (180 / pi), axis);
        trInv->SetMatrix(tr->GetMatrix());
        tr->Inverse();

        //Start!!
        double eye1[3], viewRay1[3];
        tr->TransformPoint(eye, eye1);
        tr->TransformPoint(viewRay, viewRay1);

        // Scanning
        vector<int> rr, gg, bb;
        double p_coords[3], x[3], t, rgb[3];
        //        _pc = cv::Mat(scan.res_vert * scan.res_hor, 3, CV_32FC1);
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
                    trInv->TransformPoint(x, x);
                    // data[n * 3] = x[0];
                    // data[n * 3 + 1] = x[1];
                    // data[n * 3 + 2] = x[2];
                }
                else
                {
                    trInv->TransformPoint(point, x);
                    // float xf[3] = {(float)x[0], (float)x[1], (float)x[2]};
                    // cv::Mat aRow(1, 3, CV_32FC1, xf);
                    // _pc.push_back(aRow);
                    // data[n * 3] = point[0];
                    // data[n * 3 + 1] = point[1];
                    // data[n * 3 + 2] = point[2];
                }
                float xf[3] = {(float)x[0], (float)x[1], (float)x[2]};
                cv::Mat aRow(1, 3, CV_32FC1, xf);
                _pc.push_back(aRow);
            } // Horizontal
        }     // Vertical
        timer.start();
        cv::Mat normals = cv::Mat::zeros(scan.res_vert * scan.res_hor, 3, CV_32FC1);
        float *data = (float *)normals.data;
        for (int vert = 0, n = 0; vert < scan.res_vert; vert++)
        {
            for (int hor = 0; hor < scan.res_hor; hor++, n++)
            {
                cv::Vec3d pixel = _pc.row(n);
                cv::Vec3d up, down, right, left;
                bool upChk, downChk, rightChk, leftChk;
                if (vert == 0)
                    upChk = false;
                else
                {
                    up = _pc.row((vert - 1) * scan.res_hor + hor);
                    up -= pixel;
                    upChk = fabs(up(2)) < 50;
                }

                if (vert == scan.res_vert - 1)
                {
                    downChk = false;
                }
                else
                {
                    down = _pc.row((vert + 1) * scan.res_hor + hor);
                    down -= pixel;
                    downChk = fabs(down(2)) < 50;
                }

                if (hor == 0)
                {
                    leftChk = false;
                }
                else
                {
                    left = _pc.row(vert * scan.res_hor + hor - 1);
                    left -= pixel;
                    leftChk = fabs(left(2)) < 50;
                }

                if (hor == scan.res_hor - 1)
                {
                    rightChk = false;
                }
                else
                {
                    right = _pc.row(vert * scan.res_hor + hor + 1);
                    right -= pixel;
                    rightChk = fabs(right(2)) < 50;
                }

                cv::Vec3d normal(0, 0, 0);
                if (upChk && rightChk)
                    normal += cv::normalize(right.cross(up));
                if (rightChk && downChk)
                    normal += cv::normalize(down.cross(right));
                if (downChk && leftChk)
                    normal += cv::normalize(left.cross(down));
                if (upChk && leftChk)
                    normal += cv::normalize(up.cross(left));
                normal = cv::normalize(normal);
                data[n * 3] = normal(0);
                data[n * 3 + 1] = normal(1);
                data[n * 3 + 2] = normal(2);
            }
        }
        cv::hconcat(_pc, normals, _pc);
        timer.stop();
        cout << "normal cal.: " << timer.time() << " s" << endl;
    };

    int id(0);
    function<double(vtkQuaterniond, vtkQuaterniond)> getAngle = [](vtkQuaterniond a, vtkQuaterniond b) -> double
    {
        double dotProd = a.GetW() * b.GetW() + a.GetX() * b.GetX() + a.GetY() * b.GetY() + a.GetZ() * b.GetZ();
        return 2 * acos(fabs(dotProd)) * 180 / pi;
    };
    //sample a depth
    double iso_depth = 3000;
    generateScreen(iso_depth);
    double averageS;
    int counter(0);
    cv::Vec3d isoCenter(0,0,0);
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

        cv::Mat pc_scene, pc_scene1;
        generatePC(q, pc_scene);
        cv::ppf_match_3d::writePLY(pc_scene, "scene.ply");

        vector<cv::ppf_match_3d::Pose3DPtr> results;
        timer.start();
        detectorSurf.match(pc_scene, results);
        timer.stop();
        cout << "PPF Elapsed Time " << timer.time() << " s" << endl;

        // Check results size from match call above

        size_t results_size = results.size();
        cout << "Number of matching poses: " << results_size << endl;

        if (results_size == 0)
        {
            cout << "No matching poses found. Exiting." << endl;
            exit(0);
        }

        // Get only first N results - but adjust to results size if num of results are less than that specified by N

        size_t N = 5;

        if (results_size < N)
            N = results_size;

        vector<cv::ppf_match_3d::Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

        // Create an instance of ICP

        cv::ppf_match_3d::ICP icp(100, 0.005f, 2.5f, 8);
        timer.start();

        // Register for all selected poses

        // cout << "Performing ICP on " << N << " poses..." << endl;

        icp.registerModelToScene(pc, pc_scene, resultsSub);
        timer.stop();

        cout << "ICP Elapsed Time " << timer.time() << " s" << endl;

        // Debug first five poses
        double minResidual(DBL_MAX);
        int minID(-1);
        for (size_t i = 0; i < resultsSub.size(); i++)
        {
            cv::ppf_match_3d::Pose3DPtr result = resultsSub[i];

            if (minResidual > result->residual && fabs(result->t(0)) < 500 && fabs(result->t(1)) < 500 && fabs(result->t(2)) < 500 )
            {
                minResidual = result->residual;
                minID = i;
            }
 
            cv::Mat pct = cv::ppf_match_3d::transformPCPose(pc, result->pose);
            cv::ppf_match_3d::writePLY(pct, ("result" + to_string(i) + ".ply").c_str());
            vtkQuaterniond q1(result->q(0), result->q(1), result->q(2), result->q(3));
            q1.Invert();
            cout << endl;
            cout << "residual: " << result->residual << endl;
            cout << "angle Diff. " << getAngle(q1, q) << endl;
            cout << "trans Diff. " << result->t << endl;
        }
        cout << "---------------------------------------------------------" << endl;
        if (minID < 0)
            cout << "PASS!" << endl;
        else
        {
            vtkQuaterniond q1(resultsSub[minID]->q(0), resultsSub[minID]->q(1), resultsSub[minID]->q(2), resultsSub[minID]->q(3));
            q1.Invert();
            cout << "residual: " << resultsSub[minID]->residual << endl;
            cout << "angle Diff. " << getAngle(q1, q) << endl;
            cout << "trans Diff. " << resultsSub[minID]->t << endl;
            isoCenter += resultsSub[minID]->t; counter++;
            cout << "average iso ("<<counter<<"): "<<isoCenter/counter<<endl;
        }
        cout << "=========================================================" << endl;
        getchar();
    }
    return 0;
}

cv::Mat ReadPLY(string name)
{
    cv::Mat pc;
    ifstream ifs(name);
    if (!ifs.good())
        return pc;
    vector<cv::Vec3d> vertices;
    string dump;
    int vNum, fNum;
    while (ifs >> dump)
    {
        if (dump == "vertex")
            ifs >> vNum;
        else if (dump == "face")
            ifs >> fNum;
        else if (dump == "end_header")
            break;
    }
    for (int i = 0; i < vNum; i++)
    {
        double x, y, z;
        int r, g, b;
        ifs >> x >> y >> z >> r >> g >> b;
        vertices.push_back(cv::Vec3d(x, y, z));
    }
    vector<cv::Vec3d> normals(vNum, cv::Vec3d(0, 0, 0));
    for (int i = 0; i < fNum; i++)
    {
        int tmp, a, b, c;
        ifs >> tmp >> a >> b >> c;
        cv::Vec3d normal = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
        normal = cv::normalize(normal);
        normals[a] += normal;
        normals[b] += normal;
        normals[c] += normal;
    }
    ifs.close();
    pc = cv::Mat(vNum, 6, CV_32FC1);
    float *data = (float *)pc.data;
    for (int i = 0; i < vNum; i++)
    {
        data[i * 6] = vertices[i](0);
        data[i * 6 + 1] = vertices[i](1);
        data[i * 6 + 2] = vertices[i](2);
        cv::Vec3d normal = cv::normalize(normals[i]);
        data[i * 6 + 3] = normal(0);
        data[i * 6 + 4] = normal(1);
        data[i * 6 + 5] = normal(2);
    }
    return pc;
}
