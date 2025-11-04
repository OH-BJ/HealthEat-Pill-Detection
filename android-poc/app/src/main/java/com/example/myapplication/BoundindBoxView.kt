package com.example.myapplication

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * 객체 탐지 결과를 표시하기 위해 ImageView 위에 겹쳐지는 투명한 커스텀 View입니다.
 * ImageView의 scaleType이 'fitCenter'일 때, 박스 좌표를 정확히 스케일링하여 그립니다.
 */
class BoundingBoxView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var results: List<ObjectDetectionResult> = emptyList()

    // 원본 이미지 크기 (픽셀 단위)
    private var imageWidth: Float = 0f
    private var imageHeight: Float = 0f

    // 박스 그리기 설정
    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE // 외곽선
        strokeWidth = 5f // 박스 두께
    }

    // 텍스트 그리기 설정
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
    }

    // 텍스트 배경 설정
    private val bgPaint = Paint().apply {
        color = Color.parseColor("#99000000") // 반투명 검정 배경
        style = Paint.Style.FILL
    }

    /**
     * 탐지 결과를 업데이트하고 View를 다시 그리도록 요청합니다.
     * @param results YOLO 모델에서 반환된 탐지 객체 리스트 (원본 이미지 픽셀 좌표)
     * @param originalImageWidth 탐지에 사용된 원본 이미지의 너비
     * @param originalImageHeight 탐지에 사용된 원본 이미지의 높이
     */
    fun setResults(
        results: List<ObjectDetectionResult>,
        originalImageWidth: Float,
        originalImageHeight: Float
    ) {
        this.results = results
        this.imageWidth = originalImageWidth // 예: 976.0
        this.imageHeight = originalImageHeight // 예: 1280.0
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (results.isEmpty() || imageWidth <= 0f || imageHeight <= 0f) return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        // 1. 스케일 팩터 계산 (ImageView의 scaleType='fitCenter' 기준)
        // 뷰의 크기에 이미지 크기를 나눈 비율 중 더 작은 값을 최종 스케일 팩터로 사용
        val scaleX = viewWidth / imageWidth
        val scaleY = viewHeight / imageHeight
        val scaleFactor = min(scaleX, scaleY)

        // 2. 스케일링된 이미지의 크기 계산
        val scaledImageWidth = imageWidth * scaleFactor
        val scaledImageHeight = imageHeight * scaleFactor

        // 3. 오프셋 (여백) 계산: 이미지가 중앙에 위치할 때 생기는 좌측 및 상단의 빈 공간
        // 🚨 이 오프셋을 박스 좌표에 더해야 이미지가 중앙에 배치된 만큼 박스도 이동합니다.
        val offsetX = (viewWidth - scaledImageWidth) / 2
        val offsetY = (viewHeight - scaledImageHeight) / 2

        // 🚨 여기서부터 원본 픽셀 좌표(YOLOv8에서 넘어온)를 화면 좌표로 변환합니다.
        results.forEach { result ->

            // 4. 원본 픽셀 좌표를 화면 좌표로 변환 및 스케일링
            // 화면 좌표 = (원본 픽셀 좌표 * 스케일 팩터) + 오프셋
            val left = result.xmin * scaleFactor + offsetX
            val top = result.ymin * scaleFactor + offsetY
            val right = result.xmax * scaleFactor + offsetX
            val bottom = result.ymax * scaleFactor + offsetY

            // 5. 바운딩 박스 그리기 (이하 동일)
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // 6. 텍스트 정보 그리기 (이하 동일)
            val label = "${result.className} (${String.format("%.2f", result.confidence)})"
            val textWidth = textPaint.measureText(label)
            val textRect = RectF(left, top - textPaint.textSize - 10f, left + textWidth + 10f, top)
            canvas.drawRect(textRect, bgPaint)
            canvas.drawText(label, left + 5f, top - 10f, textPaint)
        }
    }
}