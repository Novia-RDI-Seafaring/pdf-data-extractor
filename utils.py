import pdfplumber
import numpy as np
import fitz
from PIL import Image
import matplotlib.pyplot as plt
import PyPDF2
import io
import base64
import math
import json


def draw_rects(page, bboxes, color=(1, 0, 0)):
    """adds rect to page in place
    """

    shape = page.new_shape()

    for i, bbox in enumerate(bboxes):

        shape.draw_rect(bbox)
                
    shape.finish(
        #fill=(0, 0, 0),  # fill color
        color=color,  # line color
        width=0.2,  # line width
        stroke_opacity=1,  # same value for both
        fill_opacity=1,  # opacity parameters
        )

    shape.commit()

def get_page_dict(page):

    # https://pymupdf.readthedocs.io/en/latest/textpage.html#textpagedict
    textpage = page.get_textpage()
    page_dict = textpage.extractDICT()
    return page_dict
    #page.get_text('dict')

def get_text_orientation(page, rect_a):
    '''finds line in page with highest roi with given rect_a
    returns direction in (cosine, -sine)
    https://pymupdf.readthedocs.io/en/latest/textpage.html#textpagedict

    (0,1) = vertical
    (1,0) = horizontal

    '''

    textpage = page.get_textpage()
    page_dict = textpage.extractDICT()
    page_dict
    # find orientation of bbox by finding line with highest iou and then getting its direction

    highest_IOU_line = {'iou': 0}
    for i, block in enumerate(page_dict['blocks']):
        for line in block['lines']:

            line_rect = fitz.Rect(line['bbox'])
            iou = line_rect.intersect(rect_a).get_area()
            line['iou'] = iou
            if iou>highest_IOU_line['iou']:
                highest_IOU_line = line

    return highest_IOU_line['dir'], highest_IOU_line

def get_bboxs_by_searterm(page, searchTerms):
    ''' Returns bbox of texts in page that fit search term (search_for function pymupdf) and
    return super_bbox that contains all resulting bbboxes
    '''

    occurences_bbox = []

    for needle in searchTerms:
        
        occurences_bbox += page.search_for(needle)
        
    super_bbox = occurences_bbox[0]
    for bbox in occurences_bbox:
        super_bbox = super_bbox.include_rect(bbox)

    return occurences_bbox, super_bbox

def strip_dict(input_dict, keysToKeep):
    ''' Use get_page_dict to get input dict 
    '''

    stripped_dict = input_dict.copy()
    for key, value in input_dict.items():
       
        if isinstance(value, dict):
            print('is dict')
            stripped_dict[key] = strip_dict(value, keysToKeep)
        elif isinstance(value, list):
           # print('is list', key)
            for i, element in enumerate(value):
                stripped_dict[key][i] = strip_dict(element, keysToKeep)
        elif key not in keysToKeep:
            del stripped_dict[key]
    
    return stripped_dict

def pdf_to_pil_array(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pil_images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        rect = page.search_for(" ")

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pil_images.append(pil_image)

    pdf_document.close()
    
    return pil_images

def pil_to_base64(pil_image):
    # Convert PIL Image to bytes
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    # Convert bytes to base64 string
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Return base64 string in the specified format
    return f"data:image/png;base64,{base64_image}"

def convert_to_degrees(cosine, sine):
 
    # Calculate the angle in radians
    radians = math.atan2(-sine, cosine)
   
    # Convert radians to degrees
    degrees = math.degrees(radians)
   
    # Ensure the result falls within the range of 0 to 360 degrees
    if degrees < 0:
        degrees += 360
   
    return 360 - degrees

def remove_keys_recursive(d, keys_to_remove):
    """
    Recursively removes specified keys from a dictionary.
    
    Args:
    - d: The dictionary from which keys will be removed.
    - keys_to_remove: A list of keys to remove.
    
    Returns:
    - A new dictionary with specified keys removed.
    """
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if key not in keys_to_remove:
            if isinstance(value, dict):
                new_dict[key] = remove_keys_recursive(value, keys_to_remove)
            elif isinstance(value, list):
                new_dict[key] = [remove_keys_recursive(item, keys_to_remove) for item in value]
            else:
                new_dict[key] = value

    return new_dict


def readJsonFile(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return json.dumps(data)
    except FileNotFoundError:
        return "File not found."
    except json.JSONDecodeError:
        return "Error decoding JSON."
    except Exception as e:
        return f"An error occurred: {e}"