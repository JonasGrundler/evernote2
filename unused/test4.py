from docling.document_converter import DocumentConverter

source = "C:\\Users\\Jonas\\Downloads\\EnTagging\\380-2020 en.enex"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"