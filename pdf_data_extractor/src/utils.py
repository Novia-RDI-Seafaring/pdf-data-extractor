import base64
import fitz
import math
import numpy as np
import io
import json
import re

from jsonpath_ng.jsonpath import DatumInContext
from jsonpath_ng import parse
from PIL import Image

def custom_output_processor(llm_output: str, json_value: dict) -> dict:
    def recursive_find_path(datum: DatumInContext, context_paths):
        context_path = str(datum.path)
        if context_path == '$':
            return context_paths
        else:
            return recursive_find_path(datum.context, context_paths + [context_path])

    def update_dict_with_keys(dictionary, keys, value):
        current_level = dictionary
        for i, key in enumerate(keys):
            if key not in current_level:
                # add value if most inner nest
                if i == (len(keys)-1):
                    current_level[key] = value
                else:
                    current_level[key] = {}
            current_level = current_level[key]
        return dictionary
    
    expressions = [expr.strip() for expr in llm_output.split(",")]
    results = {}

    res = []
    for expression in expressions:
        exp_split = expression.split('.')
        if exp_split[-1] in ['value', 'text', 'bbox', 'dir', 'unit']:
            exp_split[-1]='*'
        expression='.'.join(exp_split)

        try:
            jsonpath_expr = parse(expression)
            found = jsonpath_expr.find(json_value)
            for datum in found:
                path = recursive_find_path(datum, [])[::-1]
                res.append((path, datum.value))
        except Exception as exc:
            raise ValueError(f"Invalid JSON Path: {expression}") from exc

    for (key_arr, value) in res:
        results = update_dict_with_keys(results, key_arr, value)
    
    return results

def extract_bboxes_and_dirs(dictionary):
    bboxes = []
    dirs = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            bbox, dir = extract_bboxes_and_dirs(value)
            bboxes.extend(bbox)
            dirs.extend(dir)
        elif key == 'bbox':
            bboxes.append(value)
        elif key == 'dir':
            dirs.append(int(convert_to_degrees(value[0], -value[1])))
   
    return bboxes, dirs
 
def extraction_wrapper(dictionary):
    '''Returns:
    - list[[x0,y0,x1,z1]] bboxes
    - degree (int) between 0-360'''
    
    bboxes, dirs = extract_bboxes_and_dirs(dictionary)
    
    im_dir = 0 if len(dirs)==0 else dirs[0]
    return bboxes, im_dir

def pdf_coords_to_img_coords(pdf_coords, pdf_height, pdf_width, im_width, im_height):
    '''PDF bbox coords as (x0, y0, x1, y1)
    to img coords in pixel coords (x0,y0, x1,y1)'''

    x0, y0, x1, y1 = pdf_coords

    x0_rel = x0/pdf_width
    x1_rel = x1/pdf_width

    y0_rel = y0/pdf_height
    y1_rel = y1/pdf_height

    rect_shape = [int(x0_rel*im_width+0.5),int(y0_rel*im_height+0.5), int(x1_rel*im_width+0.5), int(y1_rel*im_height+0.5)]

    return rect_shape

def extract_json_from_markdown(json_markdown):

    pattern = r'```json(.*?)```'
    json_string = re.search(pattern, json_markdown, re.DOTALL)
    
    return json_string.group(1).strip()

def draw_rects(page, bboxes, color=(1, 0, 0)):
    '''adds rect to page in place
    '''

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
    return base64_image
    # Return base64 string in the specified format
    #return f"data:image/png;base64,{base64_image}"

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
    '''
    Recursively removes specified keys from a dictionary.
    
    Args:
    - d: The dictionary from which keys will be removed.
    - keys_to_remove: A list of keys to remove.
    
    Returns:
    - A new dictionary with specified keys removed.
    '''
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
