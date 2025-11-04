package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import org.tensorflow.lite.support.image.ops.ResizeOp
import android.util.Log

data class ObjectDetectionResult(
    val xmin: Float,
    val ymin: Float,
    val xmax: Float,
    val ymax: Float,
    val confidence: Float,
    val classIndex: Int,
    val className: String
)

class YOLOv8Detector(
    private val context: Context,
    modelPath: String,
    private val numClasses: Int,
    private val labels: List<String>,
    private val iouThreshold: Float = 0.5f,
    private val confidenceThreshold: Float = 0.25f // 최소 신뢰도
) {

    private lateinit var tflite: Interpreter
    private val inputSize = 1280 // YOLOv8n TFLite 모델의 입력 크기

    // YOLOv8 TFLite 모델의 출력 텐서 모양 설정
    // Ultralytics export 기준 [1, num_classes + 4, num_boxes] (예: [1, 77, 8400])
    private val numBoxes = 33600
    private val outputDim = numClasses + 4 // (클래스 73개 + 박스 좌표 4개) = 77

    // 출력 텐서의 shape (batch_size, output_dim, num_boxes)
    private val outputShape = intArrayOf(1, numBoxes, outputDim)

    init {
        tflite = Interpreter(loadModelFile(context, modelPath))
    }

    private fun loadModelFile(context: Context, modelPath: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        mappedByteBuffer.order(ByteOrder.nativeOrder())
        return mappedByteBuffer
    }

    private fun calculateIoU(box1: ObjectDetectionResult, box2: ObjectDetectionResult): Float {
        val xmin = max(box1.xmin, box2.xmin)
        val ymin = max(box1.ymin, box2.ymin)
        val xmax = min(box1.xmax, box2.xmax)
        val ymax = min(box1.ymax, box2.ymax)

        // 겹치는 영역의 너비와 높이
        val intersectionW = max(0f, xmax - xmin)
        val intersectionH = max(0f, ymax - ymin)

        val intersectionArea = intersectionW * intersectionH

        // 각 박스의 넓이
        val box1Area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
        val box2Area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

        // 합집합 영역 (Union Area)
        val unionArea = box1Area + box2Area - intersectionArea

        // 0으로 나누는 것을 방지
        return if (unionArea == 0f) 0f else intersectionArea / unionArea
    }

    fun detect(bitmap: Bitmap): List<ObjectDetectionResult> {

        // 1. 입력 버퍼 준비 및 전처리 (Preprocessing)
        val inputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, inputSize, inputSize, 3),
            DataType.FLOAT32
        )
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        // 비트맵을 1280x1280으로 리사이징하고 0-1.0으로 정규화 (YOLOv8 기본 전처리)
        // 실제 전처리가 누락되어 있을 수 있으므로, 정확한 전처리 로직이 필요합니다.
        val processor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            // 🚨 수정: 255.0f로 나누는 정규화 코드를 활성화합니다. 🚨
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        val processedImage = processor.process(tensorImage)
        inputBuffer.loadBuffer(processedImage.buffer)

        // 2. 모델 출력 버퍼 준비
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        val outputs = mapOf(0 to outputBuffer.buffer)

        // 3. 모델 추론 실행
        // tflite.runForMultipleInputsOutputs(입력, 출력 맵)
        tflite.runForMultipleInputsOutputs(arrayOf(inputBuffer.buffer), outputs)

        // 4. 후처리 (Post-processing) - Channel-first [1, outputDim, numBoxes] 형태
        val outputArray = outputBuffer.floatArray
        val detections = mutableListOf<ObjectDetectionResult>()

        val originalWidth = bitmap.width.toFloat()
        val originalHeight = bitmap.height.toFloat()

        // 33600개의 박스를 반복 처리합니다.
        for (i in 0 until numBoxes) {

            // Channel-first 인덱싱: outputArray[ChannelIndex * numBoxes + BoxIndex]

            // 1. 박스 좌표 (x, y, w, h) 추출 (인덱스 0~3)
            // cx, cy, w, h는 0~1280 스케일로 정규화되어 있을 수 있습니다. (전처리 방식에 따라 다름)
            val cx = outputArray[0 * numBoxes + i]
            val cy = outputArray[1 * numBoxes + i]
            val w  = outputArray[2 * numBoxes + i]
            val h  = outputArray[3 * numBoxes + i]

            // 2. 클래스 점수 추출 및 최대 점수 찾기 (인덱스 4부터)
            var maxClassScore = 0f
            var maxClassIndex = -1

            for (j in 0 until numClasses) {
                val classChannelIndex = 4 + j
                val score = outputArray[classChannelIndex * numBoxes + i]

                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassIndex = j
                }
            }

            // 박스 신뢰도는 최대 클래스 점수
            val boxConfidence = maxClassScore

            if (boxConfidence >= confidenceThreshold && maxClassIndex != -1) {

                // 원본 픽셀 단위의 중심 좌표와 너비/높이
                val cx_pixel = cx * originalWidth
                val cy_pixel = cy * originalHeight
                val w_pixel = w * originalWidth
                val h_pixel = h * originalHeight

                // 2단계: 픽셀 단위의 Center-Width-Height를 Min-Max 좌표로 변환합니다.
                val xmin = cx_pixel - w_pixel / 2f
                val ymin = cy_pixel - h_pixel / 2f
                val xmax = cx_pixel + w_pixel / 2f
                val ymax = cy_pixel + h_pixel / 2f

                // 🚨 디버깅 로그 (이제 수백 단위의 큰 값이 나와야 합니다) 🚨
                Log.d("YOLOv8", "Box Coords: xmin=$xmin, ymax=$ymax")

                // ObjectDetectionResult 객체 생성 및 리스트에 추가
                detections.add(
                    ObjectDetectionResult(
                        xmin = xmin,
                        ymin = ymin,
                        xmax = xmax,
                        ymax = ymax,
                        confidence = boxConfidence,
                        classIndex = maxClassIndex,
                        className = labels.getOrElse(maxClassIndex) { "Unknown" }
                    )
                )
            }
        }

        // 5. NMS(Non-Maximum Suppression) 적용

        // 5-1. 신뢰도에 따라 정렬
        val sortedDetections = detections
            .filter { it.confidence >= confidenceThreshold }
            .sortedByDescending { it.confidence }
            .toMutableList()

        // 5-2. NMS 로직 실행
        val finalDetections = mutableListOf<ObjectDetectionResult>()

        while (sortedDetections.isNotEmpty()) {
            val bestDetection = sortedDetections.removeAt(0) // 가장 신뢰도 높은 박스 선택
            finalDetections.add(bestDetection)

            // 나머지 박스 중 겹치는 박스 제거
            val iterator = sortedDetections.iterator()
            while (iterator.hasNext()) {
                val current = iterator.next()

                // 같은 클래스에 대해서만 NMS 적용
                if (current.classIndex == bestDetection.classIndex) {
                    val iou = calculateIoU(bestDetection, current)
                    if (iou > iouThreshold) {
                        iterator.remove() // 겹치는 박스 제거
                    }
                }
            }
        }

        // 최종 탐지 결과 반환
        return finalDetections.toList()
    }
}