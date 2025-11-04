package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.lifecycleScope

class MainActivity : AppCompatActivity() {

    private lateinit var detector: YOLOv8Detector
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView

    private lateinit var boundingBoxView: BoundingBoxView
    private lateinit var labels: List<String>

    // 상수 설정 (사용자님의 약품 모델에 맞춤)
    private val NUM_CLASSES = 73
    private val MODEL_PATH = "best_float32.tflite"
    private val LABEL_PATH = "labels.txt"
    private val CONFIDENCE_THRESHOLD = 0.25f // 탐지 최소 신뢰도

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        boundingBoxView = findViewById(R.id.boundingBoxView)

        // 1. 레이블 로드
        labels = loadLabels(this, LABEL_PATH)
        if (labels.isEmpty()) {
            resultTextView.text = "Error: labels.txt 파일을 로드할 수 없습니다."
            return
        }

        // 2. Detector 초기화 (신뢰도 임계값 전달)
        detector = YOLOv8Detector(this, MODEL_PATH, NUM_CLASSES, labels, CONFIDENCE_THRESHOLD)

        // 3. 예시 이미지 로드 및 탐지 시작
        val sampleBitmap: Bitmap? = loadSampleBitmap(this, "sample_image.png")

        sampleBitmap?.let { originalBitmap ->
            imageView.setImageBitmap(originalBitmap)
            resultTextView.text = "모델 추론을 시작합니다..."

            lifecycleScope.launch(Dispatchers.Default) {
                // 탐지 실행
                val results = detector.detect(originalBitmap)

                withContext(Dispatchers.Main) {
                    if (results.isNotEmpty()) {
                        // 탐지된 객체가 있을 경우
                        resultTextView.text = "탐지 완료! 총 ${results.size}개 객체 탐지."

                        boundingBoxView.post {
                            boundingBoxView.setResults(
                                results,
                                originalBitmap.width.toFloat(),
                                originalBitmap.height.toFloat()
                            )
                        }

                    } else {
                        // 탐지 결과가 없거나 실패한 경우
                        resultTextView.text = "탐지된 객체가 없습니다."
                    }
                }
            }
        } ?: run {
            resultTextView.text = "샘플 이미지(sample_image.jpg)를 assets 폴더에서 로드할 수 없습니다. 파일을 확인해주세요."
        }
    }

    // MARK: - 유틸리티 함수

    // assets에서 텍스트 레이블 파일을 읽어옵니다.
    private fun loadLabels(context: Context, labelPath: String): List<String> {
        val labels = mutableListOf<String>()
        try {
            context.assets.open(labelPath).bufferedReader().useLines { lines ->
                lines.forEach { labels.add(it.trim()) }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return labels
    }

    // assets에서 샘플 이미지를 비트맵으로 로드합니다.
    private fun loadSampleBitmap(context: Context, path: String): Bitmap? {
        return try {
            context.assets.open(path).use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }


}