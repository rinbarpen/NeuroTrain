from typing import Literal, Sequence
from PIL import Image
import cv2
import numpy as np

class ImageDrawer:
    def __init__(self, image: Image.Image|cv2.Mat|np.ndarray):
        if isinstance(image, Image.Image):
            image = np.array(image)
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def draw_box(self, box: tuple[int, int, int, int], color: str = 'red', thickness: int = 2):
        """
        Draw a box on the image.

        :param box: The box coordinates (x1, y1, x2, y2).
        :param color: The color of the box.
        :param thickness: The thickness of the box.
        :return: The image with the drawn box.
        """
        cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), color, thickness)
        return self

    def draw_text(self, text: str, position: tuple[int, int], color: str = 'red', font_scale: float = 1, font_thickness: int = 2):
        """
        Draw text on the image.

        :param text: The text to draw.
        :param position: The position to draw the text (x, y).
        :param color: The color of the text.
        :param font_scale: The font scale.
        :param font_thickness: The font thickness.
        :return: The image with the drawn text.
        """
        cv2.putText(self.image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        return self

    def draw_circle(self, center: tuple[int, int], radius: int, color: str = 'red', thickness: int = 2):
        """
        Draw a circle on the image.

        :param center: The center coordinates (x, y).
        :param radius: The radius of the circle.
        :param color: The color of the circle.
        :param thickness: The thickness of the circle.
        :return: The image with the drawn circle.
        """
        cv2.circle(self.image, center, radius, color, thickness)
        return self

    def draw_point(self, point: tuple[int, int], color: str = 'red', thickness: int = 2):
        """
        Draw a point on the image.

        :param point: The point coordinates (x, y).
        :param color: The color of the point.
        :param thickness: The thickness of the point.
        :return: The image with the drawn point.
        """
        cv2.drawMarker(self.image, point, color, cv2.MARKER_CROSS, thickness)
        return self

    
    def finish_as_PIL(self) -> Image.Image:
        """
        Finish drawing and return the image.

        :return: The final image with drawn elements.
        """
        return Image.fromarray(self.image, mode='RGB')
    
    def finish_as_opencv(self) -> cv2.Mat:
        """
        Finish drawing and return the image.

        :return: The final image with drawn elements.
        """
        return self.image

    def finish(self, format: Literal['PIL', 'opencv'] = 'PIL') -> Image.Image|cv2.Mat:
        """
        Finish drawing and return the image.

        :return: The final image with drawn elements.
        """
        match format:
            case 'PIL':
                return self.finish_as_PIL()
            case 'opencv':
                return self.finish_as_opencv()
