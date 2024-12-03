import json
import os

from llama_index.core import ServiceContext
from llama_index.core.indices.struct_store import JSONQueryEngine
# from llama_index.llms import OpenAI
# from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import ImageDocument
from llama_index.core import Settings

from .single_page_pdf import SinglePagePDF
from .prompts import EXTRACT_JSON_VALUE_FROM_SCHEMA, CUSTOM_JSON_PATH_PROMPT, CUSTOM_RESPONSE_SYNTHESIS_PROMPT
from .utils import *


aoai_api_key = os.getenv('aoai_api_key')
aoai_endpoint = os.getenv('aoai_endpoint')
aoai_api_version = os.getenv('aoai_api_version')
aoai_deployment_name = os.getenv('aoai_deployment_name')

os.environ['AZURE_OPENAI_API_KEY'] = aoai_api_key
os.environ['AZURE_OPENAI_ENDPOINT'] = aoai_endpoint
os.environ['AZURE_OPENAI_API_VERSION'] = aoai_api_version
os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = aoai_deployment_name

class SearchablePDF():
    def __init__(self,
                pdf: SinglePagePDF | str,
                json_schema_string: str,
                json_value_string: str = None,
                chat_llm = None,
                multimodal_llm = None,
                embed_model = None,
                synthesize_error=False,
                verbose: bool = False,
                do_crop: bool = False) -> None:

        # First assign the parameters to instance variables
        self.chat_llm = chat_llm or AzureOpenAI(
            model="gpt-4o",
            engine="gpt-4o",  # This should match your deployment name
            temperature=0.0,
            api_key=aoai_api_key,
            api_version=aoai_api_version,
            azure_endpoint=aoai_endpoint,
            max_tokens=4000,
            azure_deployment=aoai_deployment_name,  # Add this if needed
            azure_ad_token=None,  # Add this if using Azure AD authentication
        )
        self.multimodal_llm = multimodal_llm or AzureOpenAIMultiModal(
            model="gpt-4o",
            engine="gpt-4o",
            temperature=0.0,
            api_key=aoai_api_key,
            api_version=aoai_api_version,
            azure_endpoint=aoai_endpoint,
            max_tokens=4000
        )
        self.embed_model = embed_model or AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=aoai_api_key,
            azure_endpoint=aoai_endpoint,
            api_version=aoai_api_version
        )
        
        # Then use them to configure Settings
        Settings.llm = self.chat_llm
        Settings.embed_model = self.embed_model
        
        if isinstance(pdf, str):
            pdf = SinglePagePDF(pdf_path=pdf, do_crop=do_crop)

        self.pdf = pdf
        self.multimodal_llm = multimodal_llm
        self.chat_llm = chat_llm
        self.embed_model = embed_model
        self.json_schema_string = json_schema_string
        self.synthesize_error = synthesize_error
        self.verbose = verbose

        if json_value_string is None:
            self.json_value_string = self._getText()
        else:
            self.json_value_string = json_value_string

        # TODO: add embed_model to service context
        
        # Add debug prints
        print("JSON Schema:")
        # print(json.dumps(json.loads(json_schema_string), indent=2))
        print("\nJSON Value:")
        # print(json.dumps(json.loads(self.json_value_string), indent=2))
    
        self.json_query_engine = JSONQueryEngine(
            json_value=json.loads(self.json_value_string),
            json_schema=json.loads(json_schema_string),
            llm=self.chat_llm,
            embed_model=self.embed_model,
            output_processor=custom_output_processor,
            json_path_prompt=CUSTOM_JSON_PATH_PROMPT,
            response_synthesis_prompt=CUSTOM_RESPONSE_SYNTHESIS_PROMPT,
            verbose=self.verbose
        )

        # chat history
        self.messages = []

        print("Azure endpoint:", aoai_endpoint)
        print("API version:", aoai_api_version)
        print("LLM type:", type(self.chat_llm))

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
        try:
            hidden_prompt = 'Answer as a human, either in a text paragraph or bullet points.'
            extended_query = f'{query} {hidden_prompt}'
            if self.verbose:
                print(f"Debug: Processing query: {query}")
                print(f"Debug: Extended query: {extended_query}")
            
            response = self.json_query_engine.query(extended_query)
            
            if self.verbose:
                print(f"Debug: Raw response: {response}")
                print(f"Debug: Response metadata: {response.metadata}")
            
            if not response or not hasattr(response, 'metadata') or 'json_path_response_str' not in response.metadata:
                raise ValueError("Invalid response structure")
            
            # Clean up the JSON path string to remove any explanatory text
            json_path_str = response.metadata['json_path_response_str'].strip()
            if json_path_str.startswith('```') and json_path_str.endswith('```'):
                json_path_str = json_path_str[3:-3].strip()
            
            relevant_json = custom_output_processor(json_path_str, self.json_query_engine._json_value)
            self.add_message(query, response.response)
            
            # Continue with the rest of the success path...

        except ValueError as e:
            if self.synthesize_error:
                if "Invalid JSON Path" in str(e):
                    # ... existing error handling ...
                    status = 'success'
                else:
                    status = 'failed'
            else:
                # Remove this line that's causing the default error response
                # print('default error response')
                status = 'failed'  # Change this to failed
                self.add_message(query, f"Error processing query: {str(e)}")  # More informative error message

            return {
                'status': status,
                'message_history': self.messages,
                'focus_point': None,
                'bboxes': [],
                'degrees': 0,
                'relevant_json': {}
            }

        pdf_bboxes, degrees = extraction_wrapper(relevant_json)

        (pdf_height, pdf_width) = self.pdf.dimensions
        (im_width, im_height) = self.pdf.full_size_image.size

        img_bboxes = [pdf_coords_to_img_coords(pdf_bbox, pdf_height, pdf_width, im_width, im_height) for pdf_bbox in pdf_bboxes]

        adjusted_bboxes = []
        # adjust bbox based on padding
        for bbox in img_bboxes:
            #  (x0, y0, x1, y1)
            adjust_bbox = [bbox[0]-self.pdf.left_padding, bbox[1]-self.pdf.top_padding, bbox[2]-self.pdf.left_padding, bbox[3]-self.pdf.top_padding]
            adjusted_bboxes.append(adjust_bbox)
            print('Adjust bboxes from to', bbox, adjust_bbox, )
        
        #img_bboxes = adjusted_bboxes
        if self.verbose:
            print('pdf bboxes: ', pdf_bboxes)
            print('image bboxes: ', img_bboxes)

        focus_point = None
        if len(adjusted_bboxes)>0:
            #focus point as center of first bounding box in pixels
            x = adjusted_bboxes[0][0] + (adjusted_bboxes[0][2]-adjusted_bboxes[0][0])/2
            y = adjusted_bboxes[0][1] + (adjusted_bboxes[0][3]-adjusted_bboxes[0][1])/2
            focus_point = (x, y)

        return {
            'status': 'success',
            'message_history': self.messages,
            'focus_point': focus_point, # in img coordinates
            'bboxes': adjusted_bboxes, # in img coordinates
            'degrees': degrees,
            'relevant_json': remove_keys_recursive(relevant_json, ['dir', 'bbox'])
        }
