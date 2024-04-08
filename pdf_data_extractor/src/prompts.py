from llama_index.prompts import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

EXTRACT_JSON_VALUE_FROM_SCHEMA_TEMPLATE = (
    """You are a mechanical enginner, specialized on creating, reading and understanding
    data sheets. You are able to extract all relevant textual and positional (as bounding boxes) data from the schematics and present
    them in textual format that is local and structured.
    The following is an image of a data sheet for a part.
    extract the text in it, return it as a structure json following this json schema: {json_schema_string}
    
    To help, the following represents the pdf as a json file, containing text, position and direction information for the atomic
    sections of the pdf: {page_json_string}.
    
    You separate numerical units from values into separate standard attributes but you also add a textual representation called text, 
    where value and unit are together.
    For each nested object in the JSON, provide the bounding box that outlines the property called 'text'.
    
    Organize it correctly using the given json schema.
    
    Only print the data in JSON following the provided json schema, nothing else, extract all data you find."""
)

EXTRACT_JSON_VALUE_FROM_SCHEMA = PromptTemplate(
    EXTRACT_JSON_VALUE_FROM_SCHEMA_TEMPLATE
)

CUSTOM_JSON_PATH_TMPL = (
    "We have provided a JSON schema below:\n"
    "{schema}\n"
    "Given a task, respond with JSON Path queries that "
    "can retrieve data from a JSON value that matches the schema. "
    "Seperate JSON Path queries with comma.\n"
    "Task: {query_str}\n"
    "JSONPath: "
)

CUSTOM_JSON_PATH_PROMPT = PromptTemplate(
    CUSTOM_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH
)