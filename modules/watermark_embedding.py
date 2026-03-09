import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

def embed_watermark(image, fingerprint):
    # 将彩色图像分为 R、G、B 通道
    channels = cv2.split(image)

    watermarked_channels = []
    for channel in channels:
        # 对每个通道进行 DWT 变换
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        # DCT对LL区域进行处理
        dct_LL = cv2.dct(np.float32(LL))

        # 使用QIM进行水印嵌入
        for i in range(fingerprint.shape[0]):
            for j in range(fingerprint.shape[1]):
                if fingerprint[i][j] == 1:
                    dct_LL[i][j] += 5
                else:
                    dct_LL[i][j] -= 5

        # 反DCT变换
        LL = cv2.idct(dct_LL)

        # 重构图像
        watermarked_channel = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        watermarked_channels.append(watermarked_channel)

    # 合并三个通道
    watermarked_image = cv2.merge(watermarked_channels)

    return np.uint8(watermarked_image)

def visualize_watermark_diff(original_image, watermarked_image):
    # 尺寸不一致时调整水印图像大小
    if original_image.shape != watermarked_image.shape:
        watermarked_image = cv2.resize(watermarked_image, (original_image.shape[1], original_image.shape[0]))

    # 计算每个通道（R、G、B）的差异
    diff_b = cv2.absdiff(original_image[:, :, 0], watermarked_image[:, :, 0])
    diff_g = cv2.absdiff(original_image[:, :, 1], watermarked_image[:, :, 1])
    diff_r = cv2.absdiff(original_image[:, :, 2], watermarked_image[:, :, 2])

    # 合并差异图像
    diff_image = cv2.merge([diff_b, diff_g, diff_r])

    # 显示原图、加水印图像和差异图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB))
    plt.title("Watermarked Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff_image)
    plt.title("Difference (Watermark Effect)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()