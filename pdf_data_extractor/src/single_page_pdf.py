import fitz
import numpy as np

from PIL import Image

class SinglePagePDF:

    def __init__(self, pdf_path: str, rel_page: int = 0, do_crop: bool = False) -> None:
        doc = fitz.open(pdf_path)
        self.page = doc[rel_page]
        self.page_dict = self._get_page_dict()
        self.path = pdf_path
        self.full_size_image = self.toImage()
        self.left_padding, self.right_padding, self.top_padding, self.bottom_padding = self._get_padding(pad_whitespace=do_crop)

        self.image = self._crop_padding(self.full_size_image)

    def _get_padding(self, pad_whitespace=False):
        ''' Get padding to all sides in pixels
        '''

        image = self.full_size_image#self.toImage()
        # Convert PIL Image to numpy array for easier manipulation
        image_array = np.array(image)

        height, width, _ = image_array.shape

        if not pad_whitespace:
            return  [0, 0, 0, 0]

        # Convert the image to grayscale if it's not already
        if len(image_array.shape) == 3:  # Check if it's a color image
            grayscale_image_array = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            grayscale_image_array = image_array

        # Find bounding box of non-white pixels
        non_white_indices = np.where(grayscale_image_array < 255)
        top, bottom = np.min(non_white_indices[0]), np.max(non_white_indices[0])
        left, right = np.min(non_white_indices[1]), np.max(non_white_indices[1])

        return [left.item(), (width-right).item(), top.item(), (height-bottom).item()]

    def _crop_padding(self, image: Image) -> Image:
        
        width, height = image.size
        # Crop the image using the bounding box left, upper, right, and lower 
        cropped_image = image.crop((self.left_padding, self.top_padding, width - self.right_padding, height - self.bottom_padding))
        #cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

    def _get_page_dict(self) -> dict:

        # https://pymupdf.readthedocs.io/en/latest/textpage.html#textpagedict
        textpage = self.page.get_textpage()
        page_dict = textpage.extractDICT()
        return page_dict
    
    def getRawText():
        pass

    def toImage(self) -> Image:
        ''' Coverts self.page to PIL image
        '''

        rect = self.page.search_for(" ")

        #if not rect or not do_crop:
        #    rect = self.page.rect

        pix = self.page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return pil_image

    @property    
    def dimensions(self) -> tuple[float, float]:
        return (self.page_dict['height'], self.page_dict['width'])
    
    