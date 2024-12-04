import json
import os

from llama_index.core import ServiceContext, Settings
from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import ImageDocument

from .single_page_pdf import SinglePagePDF
from .prompts import EXTRACT_JSON_VALUE_FROM_SCHEMA, CUSTOM_JSON_PATH_PROMPT, CUSTOM_RESPONSE_SYNTHESIS_PROMPT
from .utils import *

# Load environment variables once
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

        # Initialize LLM models
        self.chat_llm = chat_llm or AzureOpenAI(
            model="gpt-4o",
            engine="gpt-4o",
            temperature=0.0,
            api_key=aoai_api_key,
            api_version=aoai_api_version,
            azure_endpoint=aoai_endpoint,
            max_tokens=4000,
            azure_deployment=aoai_deployment_name,
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
        
        # Configure global settings
        Settings.llm = self.chat_llm
        Settings.embed_model = self.embed_model
        
        # Initialize other attributes
        self.pdf = SinglePagePDF(pdf_path=pdf, do_crop=do_crop) if isinstance(pdf, str) else pdf
        self.json_schema_string = json_schema_string
        self.synthesize_error = synthesize_error
        self.verbose = verbose
        self.json_value_string = json_value_string or self._getText()
        self.messages = []

        # Initialize JSON query engine
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

    def _getText(self) -> str:
        if self.verbose:
            print('generating json value')
            
        stripped_page_dict = strip_dict(self.pdf.page_dict, ['width', 'height', 'text','bbox', 'dir'])
        stripped_page_string = json.dumps(stripped_page_dict)

        response = self.multimodal_llm.complete(
            prompt=EXTRACT_JSON_VALUE_FROM_SCHEMA.format(
                json_schema_string=self.json_schema_string, 
                page_json_string=stripped_page_string
            ),
            image_documents=[ImageDocument(image=f"{pil_to_base64(self.pdf.image)}")]
        )

        try:
            json_string = extract_json_from_markdown(response.text.replace('\\n','\n'))
            if self.verbose:
                print('Finished generating json value')
            return json_string
        except Exception as e:
            raise Exception('Generated JSON value does not conform with expected format:', e)

    def add_message(self, user_query: str, bot_response: str) -> None:
        if self.verbose:
            print(f"Adding message pair - User: {user_query}, Bot: {bot_response}")
        self.messages.extend([
            {"message": user_query, "is_user": True},
            {"message": bot_response, "is_user": False}
        ])

    def query(self, query: str) -> dict:
        try:
            extended_query = f'{query} Answer as a human, either in a text paragraph or bullet points.'
            response = self.json_query_engine.query(extended_query)
            
            if not response or not hasattr(response, 'metadata') or 'json_path_response_str' not in response.metadata:
                raise ValueError("Invalid response structure")
            
            json_path_str = response.metadata['json_path_response_str'].strip()
            if json_path_str.startswith('```') and json_path_str.endswith('```'):
                json_path_str = json_path_str[3:-3].strip()
            
            relevant_json = custom_output_processor(json_path_str, self.json_query_engine._json_value)
            self.add_message(query, response.response)

        except Exception as e:
            self.add_message(query, f"Error processing query: {str(e)}")
            return {
                'status': 'failed',
                'message_history': self.messages,
                'focus_point': None,
                'bboxes': [],
                'degrees': 0,
                'relevant_json': {}
            }

        # Process bounding boxes
        pdf_bboxes, degrees = extraction_wrapper(relevant_json)
        pdf_height, pdf_width = self.pdf.dimensions
        im_width, im_height = self.pdf.full_size_image.size

        img_bboxes = [pdf_coords_to_img_coords(bbox, pdf_height, pdf_width, im_width, im_height) 
                     for bbox in pdf_bboxes]

        adjusted_bboxes = [
            [bbox[0]-self.pdf.left_padding, 
             bbox[1]-self.pdf.top_padding, 
             bbox[2]-self.pdf.left_padding, 
             bbox[3]-self.pdf.top_padding] 
            for bbox in img_bboxes
        ]

        focus_point = None
        if adjusted_bboxes:
            bbox = adjusted_bboxes[0]
            focus_point = (
                bbox[0] + (bbox[2]-bbox[0])/2,
                bbox[1] + (bbox[3]-bbox[1])/2
            )

        return {
            'status': 'success',
            'message_history': self.messages,
            'focus_point': focus_point,
            'bboxes': adjusted_bboxes,
            'degrees': degrees,
            'relevant_json': remove_keys_recursive(relevant_json, ['dir', 'bbox'])
        }
