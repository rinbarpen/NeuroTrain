from typing import Literal, Sequence
from typing_extensions import Self
from PIL import Image
import cv2
import numpy as np
import webcolors

hex_t = tuple[int, int, int]

class ImageDrawer:
    def __init__(self, image: Image.Image|cv2.Mat|np.ndarray):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, cv2.Mat):
            image = np.array(image)
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def resize(self, size: tuple[int, int]):
        self.image = cv2.resize(self.image, size)
        return self

    def draw_box(self, box: tuple[int, int, int, int], color: str|hex_t = 'red', thickness: int = 2):
        """
        Draw a box on the image.

        :param box: The box coordinates (x1, y1, x2, y2).
        :param color: The color of the box.
        :param thickness: The thickness of the box.
        :return: The image with the drawn box.
        """
        if isinstance(color, str):
            color = webcolors.name_to_rgb(color)
        cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), color, thickness)
        return self

    def draw_boxes(self, boxes: Sequence[tuple[int, int, int, int]], color: str|hex_t|Sequence[str|hex_t] = 'red', thickness: int = 2):
        """
        Draw multiple boxes on the image.

        :param boxes: The boxes to draw.
        :param color: The color of the boxes.
        :param thickness: The thickness of the boxes.
        :return: The image with the drawn boxes.
        """
        if isinstance(color, str) or isinstance(color, hex_t):
            color = [color] * len(boxes)
        for box, c in zip(boxes, color):
            self.draw_box(box, c, thickness)
        return self

    def draw_text(self, text: str, position: tuple[int, int], color: str|hex_t = 'red', font_scale: float = 1, font_thickness: int = 2):
        """
        Draw text on the image.

        :param text: The text to draw.
        :param position: The position to draw the text (x, y).
        :param color: The color of the text.
        :param font_scale: The font scale.
        :param font_thickness: The font thickness.
        :return: The image with the drawn text.
        """
        if isinstance(color, str):
            color = webcolors.name_to_rgb(color)
        cv2.putText(self.image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        return self

    def draw_texts(self, texts: Sequence[str], positions: Sequence[tuple[int, int]], color: str|hex_t|Sequence[str|hex_t] = 'red', font_scale: float = 1, font_thickness: int = 2):
        """
        Draw multiple texts on the image.

        :param texts: The texts to draw.
        :param positions: The positions to draw the texts.
        :param color: The color of the texts.
        :param font_scale: The font scale.
        :param font_thickness: The font thickness.
        :return: The image with the drawn texts.
        """
        if isinstance(color, str) or isinstance(color, hex_t):
            color = [color] * len(texts)
        for text, position, c in zip(texts, positions, color):
            self.draw_text(text, position, c, font_scale, font_thickness)
        return self

    def draw_circle(self, center: tuple[int, int], radius: int, color: str|hex_t = 'red', thickness: int = 2):
        """
        Draw a circle on the image.

        :param center: The center coordinates (x, y).
        :param radius: The radius of the circle.
        :param color: The color of the circle.
        :param thickness: The thickness of the circle.
        :return: The image with the drawn circle.
        """
        if isinstance(color, str):
            color = webcolors.name_to_rgb(color)
        cv2.circle(self.image, center, radius, color, thickness)
        return self

    def draw_circles(self, centers: Sequence[tuple[int, int]], radius: int, color: str|hex_t|Sequence[str|hex_t] = 'red', thickness: int = 2):
        """
        Draw multiple circles on the image.

        :param centers: The centers of the circles.
        :param radius: The radius of the circles.
        :param color: The color of the circles.
        :param thickness: The thickness of the circles.
        :return: The image with the drawn circles.
        """
        if isinstance(color, str) or isinstance(color, hex_t):
            color = [color] * len(centers)
        for center, c in zip(centers, color):
            self.draw_circle(center, radius, c, thickness)
        return self
    
    def draw_oval(self, center: tuple[int, int], axes: tuple[int, int], angle: int, color: str|hex_t = 'red', thickness: int = 2):
        """
        Draw an oval on the image.

        :param center: The center coordinates (x, y).   
        :param axes: The axes of the oval (x, y).
        :param angle: The angle of the oval.
        :param color: The color of the oval.
        :param thickness: The thickness of the oval.
        :return: The image with the drawn oval.
        """
        if isinstance(color, str):
            color = webcolors.name_to_rgb(color)

        cv2.ellipse(self.image, center, axes, angle, 0, 360, color, thickness)
        return self
    
    def draw_ovals(self, centers: Sequence[tuple[int, int]], axes: Sequence[tuple[int, int]], angle: int, color: str|hex_t|Sequence[str|hex_t] = 'red', thickness: int = 2):
        """
        Draw multiple ovals on the image.

        :param centers: The centers of the ovals.
        :param axes: The axes of the ovals.
        :param angle: The angle of the ovals.
        :param color: The color of the ovals.
        :param thickness: The thickness of the ovals.
        :return: The image with the drawn ovals.
        """
        if isinstance(color, str) or isinstance(color, hex_t):
            color = [color] * len(centers)
        for center, axes, c in zip(centers, axes, color):
            self.draw_oval(center, axes, angle, c, thickness)
        return self

    def draw_point(self, point: tuple[int, int], color: str = 'red', thickness: int = 2):
        """
        Draw a point on the image.

        :param point: The point coordinates (x, y).
        :param color: The color of the point.
        :param thickness: The thickness of the point.
        :return: The image with the drawn point.
        """
        if isinstance(color, str):
            color = webcolors.name_to_rgb(color)

        cv2.drawMarker(self.image, point, color, cv2.MARKER_CROSS, thickness)
        return self
    
    def draw_points(self, points: Sequence[tuple[int, int]], color: str|hex_t|Sequence[str|hex_t] = 'red', thickness: int = 2):
        """
        Draw multiple points on the image.

        :param points: The points to draw.
        :param color: The color of the points.
        :param thickness: The thickness of the points.
        :return: The image with the drawn points.
        """
        if isinstance(color, str) or isinstance(color, hex_t):
            color = [color] * len(points)
        for point, c in zip(points, color):
            self.draw_point(point, c, thickness)
        return self

    def draw_mask(self, mask: np.ndarray, color: str|hex_t = 'red', alpha: float = 0.5):
        """
        Draw a mask on the image.

        :param mask: The mask to draw. (H, W)
        :param color: The color of the mask.
        :param alpha: The alpha value for blending.
        :return: The image with the drawn mask.
        """
        mask = mask.astype(np.uint8)
        if self.image.shape != mask.shape:
            mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]))
        
        if isinstance(color, str):
            color = webcolors.name_to_rgb(color)
        color = np.array(color, dtype=np.uint8)
        mask[mask == 255] = color
        
        self.image = cv2.addWeighted(self.image, 1 - alpha, mask, alpha, 0)
        return self
    
    def draw_masks(self, masks: Sequence[np.ndarray], colors: str|hex_t|Sequence[str|hex_t] = 'red', alpha: float = 0.5) -> Self:
        """
        Draw multiple masks on the image.

        :param masks: The masks to draw. (N, H, W)
        :param colors: The colors of the masks. (N,)
        :param alpha: The alpha value for blending.
        :return: The image with the drawn masks.
        """
        if isinstance(colors, str) or isinstance(colors, hex_t):
            colors = [colors] * len(masks)
        for mask, color in zip(masks, colors):
            self.draw_mask(mask, color, alpha)
        return self


    def finish_as_PIL(self) -> Image.Image:
        """
        Finish drawing and return the image.

        :return: The final image with drawn elements.
        """
        return Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), mode='RGB')
    
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

    def save(self, output_filename: str):
        """
        Save the image to a file.

        :param output_filename: The filename to save the image.
        """
        cv2.imwrite(output_filename, self.finish_as_opencv())
