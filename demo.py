from modules.fingerprint import generate_fingerprint
from modules.watermark_embedding import embed_watermark, visualize_watermark_diff
from modules.utils import load_image, save_image

def main():
    image = load_image('image/test1.png')

    user_key = "user12345"
    user_id = 1

    # 嵌入水印并保存
    # fingerprint = generate_fingerprint(user_key, user_id)
    # print("Generated Fingerprint:\n", fingerprint)
    # watermarked_image = embed_watermark(image, fingerprint)
    # save_image(watermarked_image, 'image/watermarked_test1.png')

    # 可视化水印差异
    # watermarked_image = load_image('image/watermarked_test1.png')
    # print(image.shape)
    # print(watermarked_image.shape)
    # visualize_watermark_diff(image, watermarked_image)

if __name__ == "__main__":
    main()