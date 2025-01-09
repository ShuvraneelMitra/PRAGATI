from PyPDF2 import PdfReader
import io

# Create a PDF reader object
pdf_reader = PdfReader(io.BytesIO(b'%PDF-1.5\n%\xbf\xf7\xa2\xfe\n23 0 obj\n<< /Linearized 1 /L 174122 /H [ 1875 300 ] /O 27 /E 116465 /N 7 /T 173715 >>\nendobj\n\n24 0 obj\n<< /Type /XRef /Length 115 /Filter /FlateDecode /DecodeParms << /Columns 5 /Predictor 12 >> /W [ 1 3 1 ] /Index [ 23 173 ] /Info 21 0 R /Root 25 0 R /Size 196 /Prev 173716                /ID [<71f84e1571aa1dfdc5f1d96eed964db2><d70545c77199dd35326d6c03c66298de>] >>\nstream\nx\x9ccbd`\xe0g`b``8\t"\xd9\xfe\x82H\xc6:0\xa9\x03\x16\xcf\x07\x91\x02`Y\xb3\xf9 Ru\x1a\x884\x01\xcb*0\x81H\x95z\nendstream\nendobj\n                                                                            \nstartxref\n216\n%%EOF\n'))
# Get the number of pages
print((p))