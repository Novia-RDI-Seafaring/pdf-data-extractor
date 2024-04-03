import json
import os

from llama_index import ServiceContext
from llama_index.indices.struct_store import JSONQueryEngine
from llama_index.llms import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.schema import ImageDocument

from .single_page_pdf import SinglePagePDF
from .prompts import EXTRACT_JSON_VALUE_FROM_SCHEMA
from .utils import *

api_key = os.getenv('OPENAI_API_KEY')

class SearchablePDF():
    def __init__(self,
                 pdf: SinglePagePDF | str,
                 json_schema_string: str,
                 json_value_string: str = None,
                 chat_llm: OpenAI = OpenAI('gpt-4', max_tokens=4000, api_key=api_key),
                 multimodal_llm: OpenAIMultiModal = OpenAIMultiModal('gpt-4-vision-preview', max_new_tokens=4000, timeout=500,
                                                 image_detail='auto', api_key=api_key),
                 verbose: bool = False,
                 do_crop: bool = False) -> None:
        

        if isinstance(pdf, str):
            pdf = SinglePagePDF(pdf_path=pdf, do_crop=do_crop)

        self.pdf = pdf
        self.multimodal_llm = multimodal_llm
        self.chat_llm = chat_llm
        self.json_schema_string = json_schema_string
        self.verbose = verbose

        if json_value_string is None:
            self.json_value_string = self._getText()
        else:
            self.json_value_string = json_value_string

        self.json_query_engine = JSONQueryEngine(
            json_value=json.loads(self.json_value_string),
            json_schema=json.loads(json_schema_string),
            service_context=ServiceContext.from_defaults(llm=self.chat_llm),
            output_processor=custom_output_processor,
            verbose=self.verbose
        )

        # chat history
        self.messages = []

    def add_message(self, user_query: str, bot_response: str) -> None:
        if self.verbose:
            print("Adding user query to messages:", user_query)  # Debug print
        self.messages.append({"message": user_query, "is_user": True})  # User message
        if self.verbose:
            print("Adding bot response to messages:", bot_response)  # Debug print
        self.messages.append({"message": bot_response, "is_user": False})  # Bot response

    def _getText(self) -> str:
        if self.verbose:
            print('generating json value')
        stripped_page_dict = strip_dict(self.pdf.page_dict, ['width', 'height', 'text','bbox', 'dir'])

        stripped_page_string = json.dumps(stripped_page_dict)

        response = self.multimodal_llm.complete(
            prompt=EXTRACT_JSON_VALUE_FROM_SCHEMA.format(json_schema_string=self.json_schema_string, page_json_string=stripped_page_string),
            image_documents=[ImageDocument(image=f"{pil_to_base64(self.pdf.image)}")]
        )

        try:
            json_markdown = response.text.replace('\\n','\n')
            json_string = extract_json_from_markdown(json_markdown)
        except Exception as e:
            raise Exception('generated json value does not conform with expected format, Excpetion: ', e)

        if self.verbose:
            print('Finished generating json value')
        return json_string

    def query(self, query: str) -> dict:
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
                response = self.chat_llm.complete(prompt)

                self.add_message(query, response.text)

                status = 'success'
            else:
                pass
                # Handle other types of ValueError or re-raise if it's an unexpected error
                response = "An error occurred: " + str(e)
                # self.add_message(query, response)

                status = 'failed'

            return {
                    'status': status,
                    'message_history': self.messages,
                    'focus_point': None, # in img coordinates
                    'bboxes': [], # in img coordinates
                    'degrees': 0,
                    'relevant_json': {}
                }

        pdf_bboxes, degrees = extraction_wrapper(relevant_json)

        (pdf_height, pdf_width) = self.pdf.dimensions
        (im_width, im_height) = self.pdf.image.size

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
            'relevant_json': remove_keys_recursive(relevant_json, ['dir', 'bbox'])
        }
