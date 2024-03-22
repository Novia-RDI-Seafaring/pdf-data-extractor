#!/usr/bin/env python
# coding: utf-8

import os
import sys
import PyPDF2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import base64
import io
import base64
import requests
import re
import json
from jsonpath_ng import jsonpath, parse
from utils import *

from pdf_data_extractor import SearchablePDF

config = {
    'json_value_path': 'demo_data/he-specification.json',
    'json_schema_path': 'demo_data/he-specification_schema.json',
}




# IMPLEMENTATION OF THE GRADIO INTERFACE
import gradio as gr
import time
import os
import shutil
from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store import JSONQueryEngine
from jsonpath_ng.jsonpath import DatumInContext


#TODO move to extended query engine
def query_msg(query, searchablePDF: SearchablePDF, img_bboxes, imm):
    aug_query = f"""{query} respond nicely."""
    # Additionally, add a key 'jsonRef' that identifies the json object containing the respective information in the llama index.
    response = searchablePDF.query(aug_query)
    
    # Format and return the conversation history as before
    formatted_messages = []
    for i in range(0, len(searchablePDF.messages), 2):
        user_msg = searchablePDF.messages[i]["message"] if i < len(searchablePDF.messages) else None
        bot_msg = searchablePDF.messages[i+1]["message"] if i+1 < len(searchablePDF.messages) else None
        formatted_messages.append([user_msg, bot_msg])
    
    _, _, focus_point, bboxes, degrees, relevant_json = response.values()
    showBboxes = len(bboxes) > 0
    return createHiddenValues(bboxes, degrees), "", formatted_messages, f"```json{json.dumps(relevant_json, indent=4)}```", bboxes, show_bboxes(imm, bboxes, rotate=degrees), gr.update(visible=True), gr.update(visible=not showBboxes), gr.update(visible=showBboxes)
    #return "", formatted_messages, f"```json{json.loads(relevant_json)}```"

def upload_file(pdf_path, progress=gr.Progress()):
    try:
        progress(0, desc="Extracting data from pdf ...")
        with open(config['json_value_path']) as json_file:
            json_contents = json_file.read()            
        json_value_string = json.dumps(json.loads(json_contents))

        with open(config['json_schema_path']) as json_schema_file:
            json_schema_contents = json_schema_file.read()            
        json_schema_string = json.dumps(json.loads(json_schema_contents))

        
        searchablePDF = SearchablePDF(pdf=pdf_path, json_schema_string=json_schema_string, json_value_string=json_value_string)

        progress(0.5, desc="Making Document Searchable ...")
        
        progress(1, desc="We are ready to interact with document!")
        rel_json =  remove_keys_recursive(searchablePDF.json_query_engine._json_value, ['dir', 'bbox'])

        print(pdf_path, pdf_path.name)
        return [pdf_path, f"```json{json.dumps(rel_json, indent=4)}```", json_schema_string, searchablePDF, gr.update(visible=False), gr.update(visible=True)]
    except():
        print("done")

def get_entire_json(nl_query_engine):
    relevant_json = remove_keys_recursive(nl_query_engine._json_value, ['dir', 'bbox'])

    return f"```json{json.dumps(relevant_json, indent=4)}```" # gr.Markdown(nl_query_engine._json_value)


css = """
.markdown pre {
    height: 375px;
    padding: 10px;
}
.markdown p {
    height: 375px;
    padding: 10px;
}

#warning {
    background-color: #FFCCCB
}
.feedback textarea {
    font-size: 24px !important
}

#imageColumn .image_holder {
    transition: transform 1s ease, transform-origin 1s ease;
    transform-origin: center;
}

""" + "\n".join([f'.rotate-{deg} ' + '{ ' +f'transform: rotate({deg}deg);'+ ' }\n' for deg in range(0,360)]) + """
""" + "\n".join([f'.translate-{a}x-{b}y ' + '{ ' +f'transform: translate({a}%, {b}%);' + ' }\n' for a in range(-50, 51, 10) for b in range(-50, 51, 10)]) + """
""" + "\n".join([f'.zoom-{int(scale*10)} ' + '{ ' +f'transform: scale({scale});' + ' }\n' for scale in [round(i * 0.1, 1) for i in range(10, 51)]])

print(css)
head = """
<script>
const applyTransformations = (degrees, x, y, scaleFactor) => {
  const elements = document.querySelectorAll('#imageColumn .image_holder');
  for (element of elements) {
      console.log(degrees, x, y, scaleFactor)
      element.style.transformOrigin = `{x} {y}`;
      element.style.transform = `rotate(${degrees}deg) scale(${scaleFactor})`;
  }
};

window.imgTransform = {
    rotation: 0,
    x: 'center',
    y: 'center',
    scale: 1
}

refreshValues = () => {
    window.imgTransform.rotation = parseInt(document.getElementById('pdf_rotation').value);
    window.imgTransform.x = document.getElementById('pdf_x').value.replace('/','');
    window.imgTransform.y = document.getElementById('pdf_y').value.replace('/','');
    window.imgTransform.scale = parseFloat(document.getElementById('pdf_scale').value);
    updateImg();
}
setInterval(refreshValues, 1000);

updateImg = () => {
   const {rotation, x, y, scale} = window.imgTransform;
   applyTransformations(rotation, x, y, scale);
}


</script>
"""

def v(): pass

def createHiddenValues(bbox, dir):
    if bbox is not None and len(bbox) > 0 and len(bbox) <= 3:
        x = str(bbox[0][2] - bbox[0][0]) + "px"
        y = str(bbox[0][3] - bbox[0][1]) + "px"
        zoom = 2
    else:
        x = 'center'
        y = 'center'
        zoom = 1

    if dir is not None:
        rotation = dir
        if rotation > 180:
            rotation = rotation - 360
    else:
        rotation = 0


    return f"""
        <div id="pdfPositionValues">
            <input type="hidden" id="pdf_rotation" value={rotation}/>
            <input type="hidden" id="pdf_x" value={x}/>
            <input type="hidden" id="pdf_y" value={y}/>
            <input type="hidden" id="pdf_scale" value={zoom}/>
        </div>
        <script>refreshValues();</script>
    """

with gr.Blocks(css=css, head=head) as demo:

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

    # rotate is 0-360
    def show_bboxes(imm, img_bboxes, rotate=0):

        labels = ['label' for _ in img_bboxes]
        annotations = list(zip(img_bboxes, labels))

        print('IM annotations: ', annotations)

        return gr.update(value=(imm, annotations))
    
    json_string_relevant = gr.State([])
    img_bboxes = gr.State([])

    with gr.Row():
        with gr.Column(elem_id="imageColumn"):
            with gr.Row(visible=False, elem_classes='image_holder') as original_image_row:
                imm = gr.Image('demo_data/he-specification.jpg', show_label=False, container=False, elem_classes="originalImage")
            with gr.Row(visible=False, elem_classes='image_holder') as annotated_image_row:
                anIm = gr.AnnotatedImage(show_legend=False, show_label=False, container=False)
            with gr.Row():
                values = gr.HTML(createHiddenValues(None, None))

        with gr.Column():
            json_string_relevant = gr.Markdown(elem_classes="markdown")
            json_schema = gr.State()
            searchablePDF = gr.State()
            reset_button = gr.Button('show entire json', visible=False)
            #reset_button.click(get_entire_json, nl_query_engine, json_string_relevant)
            file_output = gr.File(visible=False, label="Upload the schema PDF")
            upload_button = gr.UploadButton("Click to Upload a File", file_types=["file"], file_count="single")
            upload_button.upload(upload_file, upload_button, [file_output, json_string_relevant, json_schema, searchablePDF, reset_button, original_image_row])

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    chat_output = gr.Chatbot()
                    chat_input = gr.Textbox(label="Enter your query")
                    chat_input.submit(query_msg, inputs=[chat_input, searchablePDF, img_bboxes,imm], outputs=[values, chat_input, chat_output, json_string_relevant, img_bboxes, anIm, reset_button, original_image_row, annotated_image_row])


if __name__ == '__main__':
    
    demo.launch()


