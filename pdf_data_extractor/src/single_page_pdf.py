import fitz
import numpy as np

from PIL import Image

class SinglePagePDF:

    def __init__(self, pdf_path, rel_page=0, do_crop=False):
        doc = fitz.open(pdf_path)
        self.page = doc[rel_page]
        self.page_dict = self._get_page_dict()
        self.path = pdf_path
        self.image = self.toImage(do_crop)

    def _crop_whitespace(self, image):

        # Convert PIL Image to numpy array for easier manipulation
        image_array = np.array(image)

        # Convert the image to grayscale if it's not already
        if len(image_array.shape) == 3:  # Check if it's a color image
            grayscale_image_array = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            grayscale_image_array = image_array

        # Find bounding box of non-white pixels
        non_white_indices = np.where(grayscale_image_array < 255)
        top, bottom = np.min(non_white_indices[0]), np.max(non_white_indices[0])
        left, right = np.min(non_white_indices[1]), np.max(non_white_indices[1])

        # Crop the image using the bounding box
        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

    def _get_page_dict(self):

        # https://pymupdf.readthedocs.io/en/latest/textpage.html#textpagedict
        textpage = self.page.get_textpage()
        page_dict = textpage.extractDICT()
        return page_dict
    
    def getRawText():
        pass

    def toImage(self, do_crop=False):
        ''' Coverts self.page to PIL image
        '''

        rect = self.page.search_for(" ")

        if not rect or not do_crop:
            rect = self.page.rect

        pix = self.page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if do_crop:
            pil_image = self._crop_whitespace(pil_image)
        
        return pil_image

    @property    
    def dimensions(self):
        return (self.page_dict['height'], self.page_dict['width'])
    
    
