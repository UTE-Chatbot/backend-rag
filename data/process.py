# import docx
# import pandas as pd
# import sys
# import os
# from unstructured.partition.auto import partition

# def extract_sections(doc):
#     sections = []
#     current_title = None
#     current_content = []
    
#     for para in doc.paragraphs:
#         if para.style.name == "Heading 1":
#             if current_title:  # Save previous section
#                 sections.append({
#                     "title": current_title,
#                     "content": "\n".join(current_content)
#                 })
#             current_title = para.text
#             current_content = []
#         else:
#             if current_title:  # Only add content if we have a title
#                 current_content.append(para.text)
    
#     # Add the last section
#     if current_title:
#         sections.append({
#             "title": current_title,
#             "content": "\n".join(current_content)
#         })
    
#     return sections

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python process.py <input_docx_file>")
#         sys.exit(1)

#     # Get absolute path of input file
#     input_file = os.path.abspath(sys.argv[1])
    
#     # Check if file exists
#     if not os.path.exists(input_file):
#         print(f"Error: File not found: {input_file}")
#         sys.exit(1)
    
#     try:
#         file_dir = os.path.dirname(input_file)
#         file_base = os.path.splitext(os.path.basename(input_file))[0]
#         output_file = os.path.join(file_dir, f"{file_base}.csv")

#         doc = docx.Document(input_file)
#         sections = extract_sections(doc)

#         # Convert sections to DataFrame and export to CSV
#         df = pd.DataFrame(sections)
#         df.to_csv(output_file, index=False, encoding='utf-8-sig')
#         print(f"Successfully exported to: {output_file}")
        
#     except docx.opc.exceptions.PackageNotFoundError:
#         print(f"Error: Unable to open Word document: {input_file}")
#         print("Make sure the file is a valid .docx file and has read permissions.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"Error: An unexpected error occurred: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


from unstructured.partition.csv import partition_csv

elements = partition_csv(filename="./data/2024_FINAL.csv")

from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(elements)

for chunk in chunks:
    print(chunk)
    print("\n\n" + "-"*80)
    input()