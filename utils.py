import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import logging
import re
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
# from langchain.document_loaders import UnstructuredPDFLoader
import re
import json
from typing import List,Tuple,Any,Union,Dict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TableExtractor:

    def __init__(self, pdf_path):
        self.huggingfacehub_api_token = "YOUR_HUGGINGFACE_API"  # Replace with your Hugging Face token
        self.repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            huggingfacehub_api_token=self.huggingfacehub_api_token,
            repo_id=self.repo_id,
            model_kwargs={"temperature": 0.1, "max_new_tokens":3000}
        )
        self.pdf_path = pdf_path

    def _image_list_(self, pdf_path: str) -> List[str]:
        """
        Converts all pages in a PDF file to images, saving them locally and returning a list of image filenames.

        Parameters:
        - pdf_path (str): The file path of the PDF document to be converted.

        Returns:
        - List[str]: A list of filenames for the images created, one per page of the PDF.

        Raises:
        - Exception: Propagates any exception that occurs during the PDF to image conversion process,
                    after logging the error.
        """

        try:
            images = convert_from_path(self.pdf_path)
            img_list = []
            for i, image in enumerate(images):
                image_name = f'page_{i}.jpg'
                image.save(image_name, 'JPEG')
                img_list.append(image_name)
            return img_list
        except Exception as e:
            logging.error(f"Error converting PDF to images: {e}")
            raise

    def _preprocess_image_(self, image_path: str) -> Any:
      """
      Preprocesses an image to enhance table detection and OCR accuracy by converting it to grayscale,
      applying noise reduction, and performing thresholding to obtain a binary image.

      Parameters:
      - image_path (str): The file path of the image to preprocess.

      Returns:
      - Any: The preprocessed image in a binary format suitable for further processing. The actual type
            is dependent on the OpenCV version used, but it generally corresponds to a numpy array.

      Raises:
      - FileNotFoundError: If the specified image file does not exist.
      - Exception: For issues related to reading the image or preprocessing steps.
      """
      try:
        img = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        # Thresholding to get a binary image
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

      except Exception as e:
          logging.error("Error during the preprocessing of the image", exc_info=True)
          raise


    def _detect_tables_(self, image: Any) -> List[Tuple[int, int, int, int]]:
      """
      Detects tables in an image using morphological transformations for line detection
      and contour detection to identify table boundaries.

      Parameters:
      - image (Any): The preprocessed binary image where tables are to be detected. The type is
                    typically a NumPy array, though it is annotated as `Any` to accommodate for
                    flexibility in input image types.

      Returns:
      - List[Tuple[int, int, int, int]]: A list of tuples, each representing the bounding box of a detected
                                        table in the format (x, y, width, height).

      Note:
      This method assumes the input image is preprocessed, ideally binary, to highlight table structures.
      """
      try:
        # Use morphological transformations to detect lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image.shape[0] / 30)))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[1] / 30), 1))
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

        # Combine lines
        table_grid = cv2.add(horizontal_lines, vertical_lines)
        # Find contours
        contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > image.size * 0.001:  # Filter out small contours
                tables.append((x, y, w, h))

        logging.info(f"Detected {len(tables)} tables in the image.")
        return tables

      except Exception as e:
        logging.error("Error during table detection", exc_info=True)
        raise


    def _extract_text_from_tables_(self, image: Any, tables: List[Tuple[int, int, int, int]]) -> List[str]:
      """
      Extracts text from specified table regions in an image using OCR.

      Parameters:
      - image (Any): The image from which text is to be extracted. The type is typically a NumPy array,
                    though it is annotated as `Any` to accommodate for flexibility in input image types.
      - tables (List[Tuple[int, int, int, int]]): A list of tuples, each representing the bounding box
                                                  of a table to extract text from, in the format
                                                  (x, y, width, height).

      Returns:
      - List[str]: A list of strings, where each string contains the text extracted from the corresponding
                  table region defined in the `tables` parameter.

      Raises:
      - Exception: For issues during the image cropping or OCR process.
      """
      try:
        texts = []
        for (x, y, w, h) in tables:
            table_image = image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(table_image, lang='eng')
            texts.append(text)
        logging.info(f"Extracted text from {len(tables)} tables.")
        return texts
      except Exception as e:
        logging.error("Error extracting text from tables", exc_info=True)
        raise


    def extract_tables_and_text(self) -> List[str]:
      """
      Extracts tables and their respective text from the document specified by `self.pdf_path`.

      This method integrates the workflow of converting PDF pages to images, preprocessing images for table
      detection, detecting table boundaries, and extracting text from these tables.

      Returns:
      - List[str]: A list of strings, each string contains the text extracted from a table detected in the
                  document. The list is compiled from all tables detected across all pages of the document.

      Raises:
      - Exception: For any issues encountered during the processes of image conversion, preprocessing,
                  table detection, or text extraction.
      """
      try:
        logging.info("Starting table and text extraction process.")
        # Convert all pages of the PDF to images and store the paths in `images`.
        images = self._image_list_(self.pdf_path)

        # Initialize an empty list to hold all extracted texts from tables.
        all_tables_text = []

        # Iterate through each image path in the list of images.
        for image_path in images:
            preprocessed_image = self._preprocess_image_(image_path)
            tables = self._detect_tables_(preprocessed_image)
            texts = self._extract_text_from_tables_(preprocessed_image, tables)
            all_tables_text.extend(texts)

        logging.info("Completed table and text extraction process.")
        # Return the list of extracted texts from all tables.
        return all_tables_text
      except Exception as e:
        logging.error("Error in extracting tables and text", exc_info=True)
        raise

    def extracted_data(self) -> List[str]:
      """
      Cleans and returns the extracted text data from tables in the document.

      This method calls `extract_tables_and_text` to get the raw text from tables,
      then cleans the text by normalizing spaces and removing excessive newlines.

      Returns:
          List[str]: A list of cleaned strings, each representing the text extracted
                    and cleaned from a single table detected in the document.
      """
      try:
        # Log the start of the data extraction process
        logging.info("Starting extracted data processing.")

        # Extract raw tables text
        tables_text = self.extract_tables_and_text()

        # Initialize an empty list to hold cleaned text data
        answer=[]

        # Iterate through each raw text extracted from tables
        for text in tables_text:
          # Replace multiple spaces or tabs with a single space
          cleaned_string = re.sub(r'[ \t]+', ' ', text)
          cleaned_string = re.sub(r'\n\s*\n', '', cleaned_string)
          answer.append(cleaned_string)
        logging.info("Completed data extraction and cleaning.")
        return answer

      except Exception as e:
          logging.error("Error in extracting data", exc_info=True)
          raise

    def response(self, content: str) -> str:
      """
      Processes the given content by formatting it into a key-value pair JSON-like structure using an AI assistant.

      Args:
          content (str): The input data that needs to be analyzed and formatted.

      Returns:
          str: The cleaned and formatted result as a JSON-like string, where keys without values are set to an empty string.
      """

      try:

        # Define the template for processing the input content
        template = """[INST]you are json formatter.your task analyze the given data{data} and must return answer as json.key doest have value return empty string.only generate json for given data's.all answers should be in  json format(for all data).[/INST]"""
        # Assuming PromptTemplate and LLMChain are correctly defined and imported
        prompt = PromptTemplate(template=template, input_variables=["data"])
        llm_chain = LLMChain(prompt=prompt, verbose=True, llm=self.llm)
        # Run the language model chain with the provided content
        result = llm_chain.run({"data":content})
        # # Clean the result by removing the pattern specified
        # pattern = r"\*[^*]*\*"
        # cleaned_text = re.sub(pattern, '', result)
        # # Log the completion of the cleaning process
        # logging.info("Completed processing and cleaning the response.")
        # print("result",result)
        return result

      except Exception as e:
          logging.error("Error in response", exc_info=True)
          raise

    def list_of_answer(self) -> List[str]:
      """
      Processes extracted data to generate a list of answers after further cleaning and formatting.

      This method iterates over the data extracted by `extracted_data`, processes each item using `response`,
      and compiles the results into a final list.

      Returns:
          List[str]: A list of strings, each a processed and cleaned response based on the extracted data.
      """
      try:
        # Retrieve extracted data
        answer=self.extracted_data()
        # Initialize an empty list to hold the final processed results
        final=[]
        # Iterate over each item in the extracted data
        for i in range(len(answer)):
          result=self.response(answer[i])
          final.append(result)
        logging.info("Completed processing list of answers.")
        return final

      except Exception as e:
          logging.error("Error in list of answer", exc_info=True)
          raise

    def extract_and_combine_json(self,text_list: List[str]) -> List[Dict[str, Any]]:
      """
      Extracts JSON objects from a list of strings and combines them into a single list.

      Each string in the input list is searched for JSON objects enclosed within ```json ... ``` markers.
      All found JSON objects are parsed and combined into a list of dictionaries.

      Args:
          text_list: A list of strings, each potentially containing one or more JSON objects.

      Returns:
          A list of dictionaries, where each dictionary is a parsed JSON object found in the input text.

      Note:
          This function uses a specific pattern to identify JSON blocks within the text, which are enclosed in
          triple backticks followed by 'json' keyword and assumes well-formed JSON objects.
      """
      try:
        # This pattern matches your JSON blocks specifically formatted in your example
        pattern = r'```json\n({.*?})\n```'
        combined_json_objects = []  # This will hold all your parsed JSON objects

        for text in text_list:
            # Find all JSON strings within the text
            json_strings = re.findall(pattern, text, re.DOTALL)
            for json_str in json_strings:
                try:
                    # Parse the JSON string and append the resulting object to your list
                    json_obj = json.loads(json_str)
                    combined_json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")
        return combined_json_objects

      except Exception as e:
        print(f"Error in extract_and_combine_json {e}")

    def key_value_pair(self) -> str:
      """
      Extracts JSON objects from a list of text blocks, combines them, and returns the combined JSON as a string.

      This method calls `list_of_answer` to retrieve a list of text blocks, each potentially containing JSON objects.
      These blocks are then processed by `extract_and_combine_json` to extract and combine all JSON objects into a single structure.
      Finally, it converts this structure into a nicely formatted JSON string.

      Returns:
          A string representation of the combined JSON objects, formatted with an indent of 2 spaces.
      """
      try:
        # Retrieve the list of text blocks that may contain JSON objects
        list_of_text=self.list_of_answer()
        # Extract and combine JSON objects from the text blocks
        combined_json=self.extract_and_combine_json(list_of_text)
        # Convert the combined JSON objects into a formatted string
        key_value=json.dumps(combined_json, indent=2)
        logging.info("Successfully combined JSON objects.")
        return key_value

      except Exception as e:
          logging.error(f"An error occurred in key_value_pair: {e}")
          raise
