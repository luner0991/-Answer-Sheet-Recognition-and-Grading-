'''
@Author: lxy
@Function: 答题卡识别判卷
@Date: 2025/1/25
@Email: xliu80036@gmail.com
'''
import numpy as np
import cv2
import imutils  # imutils库，用于简化OpenCV操作

# 正确答案字典，键为题目编号，值为正确答案的选项编号，自行设置
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

def order_points(pts):
    """
    对4个点进行排序，顺序为：左上、右上、右下、左下。
    :param pts: 输入的4个点坐标。
    :return: 排序后的4个点坐标。
    """
    rect = np.zeros((4, 2), dtype="float32")  # 初始化一个4x2的矩阵，用于存储排序后的点
    s = pts.sum(axis=1)  # 对每个点的x和y坐标求和
    rect[0] = pts[np.argmin(s)]  # 左上角是x+y最小的点
    rect[2] = pts[np.argmax(s)]  # 右下角是x+y最大的点
    diff = np.diff(pts, axis=1)  # 计算每个点的x和y坐标的差值
    rect[1] = pts[np.argmin(diff)]  # 右上角是x-y最小的点
    rect[3] = pts[np.argmax(diff)]  # 左下角是x-y最大的点
    return rect  # 返回排序后的4个点

def four_point_transform(image, pts):
    """
    对图像进行透视变换，将倾斜的答题卡转换为正视图。
    :param image: 输入图像。
    :param pts: 答题卡的4个顶点坐标。
    :return: 透视变换后的图像。
    """
    rect = order_points(pts)  # 对4个点进行排序
    (tl, tr, br, bl) = rect  # 分别获取左上、右上、右下、左下角的点

    # 计算新的宽度和高度
    widthA = np.linalg.norm(br - bl)  # 计算底边的宽度
    widthB = np.linalg.norm(tr - tl)  # 计算顶边的宽度
    maxWidth = max(int(widthA), int(widthB))  # 取最大宽度

    heightA = np.linalg.norm(tr - br)  # 计算右边的高度
    heightB = np.linalg.norm(tl - bl)  # 计算左边的高度
    maxHeight = max(int(heightA), int(heightB))  # 取最大高度

    # 定义变换后的目标坐标
    dst = np.array([
        [0, 0],  # 左上角
        [maxWidth - 1, 0],  # 右上角
        [maxWidth - 1, maxHeight - 1],  # 右下角
        [0, maxHeight - 1]  # 左下角
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped  # 返回变换后的图像

def sort_contours(cnts, method="left-to-right"):
    """
    对轮廓进行排序。
    :param cnts: 轮廓列表。
    :param method: 排序方法（从左到右、从右到左、从上到下、从下到上）。
    :return: 排序后的轮廓和对应的边界框。
    """
    reverse = method in ["right-to-left", "bottom-to-top"]  # 判断是否需要反转排序
    i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0  # 选择排序依据的坐标轴
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 计算每个轮廓的边界框
    # 对轮廓和边界框进行排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes  # 返回排序后的轮廓和边界框

def preprocess_image(image):
    """
    图像预处理：灰度化、高斯模糊、边缘检测。
    :param image: 输入图像。
    :return: 边缘检测结果。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 对灰度图进行高斯模糊
    edged = cv2.Canny(blurred, 75, 200)  # 对模糊后的图像进行边缘检测
    return edged  # 返回边缘检测结果

def find_document_contour(edged):
    """
    查找答题卡的轮廓。
    :param edged: 边缘检测结果。
    :return: 答题卡的4个顶点坐标。
    """
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    cnts = imutils.grab_contours(cnts)  # 兼容不同OpenCV版本的轮廓提取
    if len(cnts) == 0:  # 如果没有找到轮廓
        raise ValueError("无法找到轮廓！")  # 抛出异常

    # 按轮廓大小排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 遍历轮廓，找到矩形轮廓
    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 对轮廓进行多边形近似
        if len(approx) == 4:  # 如果近似结果为4个点，则认为是答题卡轮廓
            return approx.reshape(4, 2)  # 返回4个顶点坐标
    raise ValueError("无法找到答题卡轮廓！")  # 如果没有找到4个点的轮廓，抛出异常

def grade_exam(warped, answer_key):
    """
    对答题卡进行判卷。
    :param warped: 透视变换后的图像。
    :param answer_key: 正确答案。
    :return: 判卷后的图像和分数。
    """
    # 对图像进行二值化处理
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)  # 显示二值化结果
    cv2.waitKey(0)

    # 查找选项轮廓
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    questionCnts = []  # 存储符合条件的选项轮廓
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)  # 计算轮廓的边界框
        ar = w / float(h)  # 计算宽高比
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:  # 过滤掉不符合条件的轮廓
            questionCnts.append(c)

    # 按从上到下排序
    questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0  # 初始化正确答案数量
    # 将图像转换为彩色图像（BGR）
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # 每排5个选项
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = sort_contours(questionCnts[i:i + 5])[0]  # 对每排的5个选项排序
        bubbled = None  # 初始化当前排的填充选项

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")  # 创建掩码
            cv2.drawContours(mask, [c], -1, 255, -1)  # 在掩码上绘制轮廓
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # 与二值图像进行与操作
            total = cv2.countNonZero(mask)  # 计算非零像素数量

            if bubbled is None or total > bubbled[0]:  # 找到填充最多的选项
                bubbled = (total, j)

        k = answer_key[q]  # 获取正确答案
        color = (0, 255, 0) if k == bubbled[1] else (0, 0, 255)  # 绿色表示正确，红色表示错误
        if k == bubbled[1]:  # 如果用户选择正确
            correct += 1  # 增加正确答案数量
        cv2.drawContours(warped, [cnts[k]], -1, color, 3)  # 在图像上绘制结果

    # 计算分数
    score = (correct / len(answer_key)) * 100
    return warped, score  # 返回判卷后的图像和分数

def main():
    # 读取图像,可自行修改测试图像地址
    image_path = "images/test_03.png"
    image = cv2.imread(image_path)
    if image is None:  # 如果图像加载失败
        print("[ERROR] 图像加载失败，请检查路径是否正确！")
        return

    # 显示原始图像
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # 预处理图像
    edged = preprocess_image(image)
    cv2.imshow("Edged Image", edged)  # 显示边缘检测结果
    cv2.waitKey(0)

    # 查找答题卡轮廓
    try:
        docCnt = find_document_contour(edged)
    except ValueError as e:
        print(f"[ERROR] {e}")  # 如果找不到轮廓，输出错误信息
        return

    # 绘制轮廓
    contours_img = image.copy()
    cv2.drawContours(contours_img, [docCnt], -1, (0, 255, 0), 3)  # 在图像上绘制轮廓
    cv2.imshow("Contours", contours_img)  # 显示轮廓图像
    cv2.waitKey(0)

    # 透视变换
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    warped = four_point_transform(gray, docCnt)  # 对图像进行透视变换

    # 判卷
    graded_image, score = grade_exam(warped, ANSWER_KEY)
    cv2.imshow("graded_image", warped)  # 显示判卷后的图像
    cv2.waitKey(0)

    # 显示结果
    print(f"[INFO] 分数: {score:.2f}")  # 输出分数
    cv2.putText(graded_image, f"{score:.2f}", (10, 30),  # 在图像上绘制分数
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Exam Result", graded_image)  # 显示最终结果
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 关闭所有窗口

if __name__ == "__main__":
    main()  # 运行主函数