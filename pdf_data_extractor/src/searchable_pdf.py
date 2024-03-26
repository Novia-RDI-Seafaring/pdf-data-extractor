from .single_page_pdf import SinglePagePDF
from .prompts import EXTRACT_JSON_VALUE_FROM_SCHEMA

from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store import JSONQueryEngine
from llama_index.core.llms.types import ChatMessage
import json
import os
from .utils import *

api_key = os.getenv('OPENAI_API_KEY')

class SearchablePDF():

    def __init__(self, 
                 pdf: SinglePagePDF | str,
                 json_schema_string: str,
                 json_value_string = None,
                 api_key=None,
                 chat_llm=OpenAI('gpt-4', max_tokens=4000, api_key=api_key),
                 vision_llm=OpenAI('gpt-4-vision-preview', max_tokens=4000, api_key=api_key),
                 verbose=False):
        

        if isinstance(pdf, str):
            pdf = SinglePagePDF(pdf_path=pdf)

        self.pdf = pdf
        self.vision_llm = vision_llm
        self.json_schema_string = json_schema_string
        self.verbose = verbose

        if json_value_string is None:
            self.json_value_string = self._getText()
        else:
            self.json_value_string = json_value_string
        
        self.json_query_engine = JSONQueryEngine(
            json_value=json.loads(self.json_value_string),
            json_schema=json.loads(json_schema_string),
            service_context=ServiceContext.from_defaults(llm=chat_llm),
            output_processor=custom_output_processor,
            verbose=self.verbose
        )

        # chat history
        self.messages = []
    
    def add_message(self, user_query, bot_response):
        if self.verbose:
            print("Adding user query to messages:", user_query)  # Debug print
        self.messages.append({"message": user_query, "is_user": True})  # User message
        if self.verbose:
            print("Adding bot response to messages:", bot_response)  # Debug print
        self.messages.append({"message": bot_response, "is_user": False})  # Bot response

    def _getText(self):
        page_dict = strip_dict(self.pdf.page_dict, ['width', 'height', 'text','bbox', 'dir'])

        page_json_string = json.dumps(page_dict)
        
        response = self.vision_llm.chat(
            messages=[
                ChatMessage(role='user', content=[
                {
                "type": "text",
                "text": EXTRACT_JSON_VALUE_FROM_SCHEMA.format(json_schema_string=self.json_schema_string, page_json_string=page_json_string)
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": pil_to_base64(self.pdf.toImage())
                }
                }
            ])
            ]
        )

        json_string = extract_json_from_markdown(response)
        
        return json_string

    def query(self, query):
        '''
        '''
        try:
            response = self.json_query_engine.query(query)
            relevant_json = custom_output_processor(response.metadata['json_path_response_str'], self.json_query_engine._json_value)
            
            self.add_message(query, response.response)

            if self.verbose:
                print('Relevant json: -----', relevant_json)
                print("Current conversation history:", self.messages)

        # TODO handle exceptions
        except ValueError as e:
            # Handle cases where the query does not match the schema
            if "Invalid JSON Path" in str(e):
                print('------------ Exception that was thrown: ', e)
                prompt = f"""The user asked: '{query}', which was not found by the JsonQueryEngine. Use the following error to provide
                a helpful response: {e}."""
                #print('prompt to provide good error message: ', prompt)

                #response = query_gpt4(prompt)
                # response = "This information is not in the JSON Schema. Please ask for details within the specification."
                
                #nl_query_engine.add_message(query, response),
                
            else:
                pass
                # Handle other types of ValueError or re-raise if it's an unexpected error
                #response = "An error occurred: " + str(e)
                #nl_query_engine.add_message(query, response)

            return {
                    'status': 'failed',
                    'message_history': self.messages,
                    'focus_point': None, # in img coordinates
                    'bboxes': None, # in img coordinates
                    'degrees': 0,
                    'relevant_json': {}
                }
        
        pdf_bboxes, degrees = extraction_wrapper(relevant_json)
        
        (pdf_height, pdf_width) = self.pdf.dimensions
        
        # TODO remove hard coded
        im_width = 1684 
        im_height = 1191

        img_bboxes = [pdf_coords_to_img_coords(pdf_bbox, pdf_height, pdf_width, im_width, im_height) for pdf_bbox in pdf_bboxes]
        
        if self.verbose:
            print('pdf bboxes: ', pdf_bboxes)
            print('image bboxes: ', img_bboxes)

        x = str(img_bboxes[0][2] - img_bboxes[0][0])
        y = str(img_bboxes[0][3] - img_bboxes[0][1])

        return {
            'status': 'success',
            'message_history': self.messages,
            'focus_point': (x, y), # in img coordinates
            'bboxes': img_bboxes, # in img coordinates
            'degrees': degrees,
            'relevant_json': relevant_json
        }


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

    pdf_path = os.path.join(base_path, 'demo_data/he-specification.pdf')
    json_schema_path = os.path.join(base_path, 'demo-data/he-specification_schema.json')
    json_schema_string = readJsonFile(json_schema_path)

    searchablePDF = SearchablePDF(pdf=pdf_path, json_schema_string=json_schema_string)
    response = searchablePDF.query('What is the max temp of side 1?')