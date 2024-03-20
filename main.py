from utils import TableExtractor
if __name__=="__main__":
  pdf_path="YOUR_PDF_PATH"
  table=TableExtractor(pdf_path)
  result=table.key_value_pair()
  print(result)
