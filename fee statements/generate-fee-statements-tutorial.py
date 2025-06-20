print('Fee Statements Generator') #This script generates fee statements in PDF format for students based on their enrollment and payment status

#make sure you have the required libraries installed
#run the following command in your terminal
#pip install fpdf
from fpdf import FPDF

#add your institution's logo and other details
#you can use the FPDF library to add images, text, and other elements to the PDF
class PDF(FPDF):
    def header(self):
        #logo
        #select the image file you want to use as a logo and adjust the size and position
        self.image('fee statements/logo_used_as_example.png', 10, 8, 30) 
        #just set the width, height will be adjusted automatically to maintain aspect ratio
        #hence preventing distortion or stretching of the image
        #set the font for the header
        #specify font
        #fonts may vary based on your system and your liking
        #download the font you want to use
        # 'B' for bold, 'I' for italic, 'U' for underline
        self.set_font('Arial', 'B', 25)
        #add a cell to the top central part of the page
        #set the width of the cell to 0, so it will take the full width of the page
        #Ln=True or Ln=1 means the cell will be followed by a new line
        #if you use ln=False or ln=0, the next cell will be on the same line
        #set the alignment to 'C' for center
        self.cell(0, 50, 'Institution Name!', ln=True, align='C' , border=False)
        #add a line break
        self.ln(20)

# Create a PDF object and generate the PDF
pdf = PDF(format='A4', unit='mm')
#set auto page break
pdf.set_auto_page_break(auto=True, margin=15) #margin is the distance from the bottom of the page
pdf.add_page()
pdf.output('fee_statement_new.pdf')
