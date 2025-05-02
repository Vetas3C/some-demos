#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 640


// COCO类别（可以简化，只保留常用），只有在这里面的类别会被在图片上标注出来，这里只列出来了COCO数据集的0-10，一共11个类
const vector<string> class_names = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant"
    // 这里可以继续补充完整，或者只用你需要的
};

int main() {
    // 加载ONNX模型
    Net net = readNetFromONNX("yolov5s.onnx");
	
	//设置推理后端
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);  // 或者DNN_BACKEND_OPENCV
    net.setPreferableTarget(DNN_TARGET_CPU);        // 嵌入式板子默认用CPU

    // 读取输入图片
    Mat image = imread("test.jpg");
    if (image.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }

    // 预处理，转换为输入张量blob（注意大小要和训练时一致，默认640x640）
    Mat blob = blobFromImage(image, 1/255.0, Size(IMAGE_WIDTH, IMAGE_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    // 对图片进行推理，得到输出outputs，这是一个形状为 [1, N, 85] 的张量
	//N：检测到的目标数量（比如25200个anchor）
	//85 = 4坐标 + 1置信度 + 80类别得分，，也就是对于每个检测到的目标，它们都有一个向量[x, y, w, h, conf, 80 classes]
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 解析输出
    float confThreshold = 0.5;  // 置信度阈值
    float nmsThreshold = 0.4;   // NMS抑制阈值

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    Mat output = outputs[0]; //outputs[0]就是上面的[N, 85]
    const int dimensions = output.size[2];
    const int rows = output.size[1];

    float* data = (float*)output.data;//将输出传递给float*类型的变量data中，便于后面使用

	//遍历每一个检测到的目标
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];//检查置信度，如果小于0.5就丢弃
        if (confidence >= confThreshold) {
            float* classes_scores = data + 5;//data+5就是存放各个类的置信度的地址
            Mat scores(1, class_names.size(), CV_32FC1, classes_scores);//查找我们定义在class_names里面的类型的对应的置信度得分，因为class_names.size()为11，所以其实就是获取了outputs的前11个类型的置信度
            Point classIdPoint;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);//minMaxLoc找到最大类别得分，以及对应的类别ID

            if (max_class_score > confThreshold) {
                // 坐标还原到原图尺寸
                float cx = data[0] * image.cols;
                float cy = data[1] * image.rows;
                float w  = data[2] * image.cols;
                float h  = data[3] * image.rows;
                int left = int(cx - w/2);
                int top = int(cy - h/2);
				
				//记录下通过筛选的检测框
                classIds.push_back(classIdPoint.x);//类别ID
                confidences.push_back(confidence);//置信度
                boxes.push_back(Rect(left, top, int(w), int(h)));//框的位置
            }
        }
        data += dimensions;
    }

    // NMS后处理，统一做NMS（非极大值抑制）。
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	//画框 + 标注
    for (int idx : indices) {
        Rect box = boxes[idx];
        rectangle(image, box, Scalar(0, 255, 0), 2);

        string label = format("%.2f", confidences[idx]);
        if (!class_names.empty()) {
            label = class_names[classIds[idx]] + ": " + label;
        }
        int baseline;
        Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        putText(image, label, Point(box.x, box.y - label_size.height),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

    // 保存或展示结果
    imwrite("result.jpg", image);
    cout << "Detection finished, result saved to result.jpg" << endl;

    return 0;
}
