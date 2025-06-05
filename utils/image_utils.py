from typing import Literal
from PIL import Image, ImageDraw
import logging
from pathlib import Path
import numpy as np

class ImageUtils:

    @staticmethod
    def draw_bounding_boxes(
        image: Image.Image, boxes: list, color: str = "red", thickness: int = 2
    ) -> Image.Image:
        """
        Draw bounding boxes on the image.

        :param image: The input image.
        :param boxes: List of bounding boxes in the format [x1, y1, x2, y2].
        :param color: Color of the bounding box.
        :param thickness: Thickness of the bounding box lines.
        :return: Image with bounding boxes drawn.
        """
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(box, outline=color, width=thickness)
        return image

    @staticmethod
    def split(
        image: Image.Image,
        n_rows: int = 1,
        n_cols: int = 1,
        *,
        output_dir: Path | None = None,
    ) -> list[Image.Image]:
        """
        将图像分割成小块并保存到指定目录。

        Args:
            image (Image.Image): 图像文件实例。
            n_rows (int): 一行图像块的数量。
            n_cols (int): 一列图像块的数量。
            output_dir (Path): 保存小块图像的目录。
        """

        width, height = image.size
        tile_width, tile_height = width // n_cols, height // n_rows

        tile_images = []
        for i in range(0, height, tile_height):
            for j in range(0, width, tile_width):
                # 定义当前小块的边界
                box = (j, i, j + tile_height, i + tile_width)

                # 避免超出图像边界
                if box[2] > width:
                    box = (box[0], box[1], width, box[3])  # Adjust right boundary
                if box[3] > height:
                    box = (box[0], box[1], box[2], height)  # Adjust bottom boundary

                # 提取小块
                try:
                    tile_image = image.crop(box)
                except Exception as e:
                    logging.error(f"切除图像失败。边界：{box}，错误：{e}")
                    raise e

                tile_images.append(tile_image)

        if output_dir:
            for i, tile_image in enumerate(tile_images):
                output_filename = output_dir / f"tile_{i:04d}.png"
                tile_image.save(output_filename)

        return tile_images

    @staticmethod
    def region_upsample(
        image: Image.Image,
        region: tuple[int, int, int, int],
        resize: tuple[int, int] | None = None,
        region_shape: Literal['rectangle', 'circle']='rectangle',
        *,
        outline: str='red',
        width: int=2,
        output_dir: Path | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Image region upsampling.
        :param image: The input image.
        :param region: The region to crop, defined as (x1, y1, x2, y2).
        :param resize: The size to resize the cropped region to.
        :param region_shape: The shape of the region to draw ("rectangle" or "circle").
        :param outline: The color of the outline.
        :param width: The width of the outline.
        :param output_dir: The directory to save the output images.
        :return: A tuple containing the cropped image and the image with the drawn region.
        """
        region_image = image.copy()
        draw = ImageDraw.Draw(region_image)
        if region_shape == "circle":
            if region[2] - region[0] != region[3] - region[1]:
                raise ValueError("For circle, the width and height of the region must be equal.")
            center_xy = region[0] + (region[2] - region[0]) // 2, region[1] + (region[3] - region[1]) // 2
            radius = (region[2] - region[0]) // 2
            draw.circle(center_xy, radius, outline=outline, width=width)

            crop_image = image.crop(region)
            mask = Image.new('L', crop_image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, *(crop_image.size)), fill=255)
            crop_image.putalpha(mask)
        else:
            draw.rectangle(region, outline=outline, width=width)
            crop_image = image.crop(region)

        if resize:
            crop_image = crop_image.resize(resize)
        if output_dir:
            crop_image.save(output_dir / "crop_image.png", "PNG")
            region_image.save(output_dir / "region_image.png", "PNG")

        return crop_image, region_image


    @staticmethod
    def to_numpy(image: Image.Image, dtype: np.dtype = np.float32):
        """
        Convert a PIL Image to a NumPy array.

        :param image: The input image.
        :param dtype: The desired NumPy data type.
        :return: NumPy array representation of the image.
        """
        return np.array(image, dtype=dtype)
