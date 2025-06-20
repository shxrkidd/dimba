from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('fee statements/logo_used_as_example.png', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Fee Statement', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def student_details(self, name, adm_no, course):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f'Student Name: {name}', 0, 1)
        self.cell(0, 10, f'Admission No: {adm_no}', 0, 1)
        self.cell(0, 10, f'Course: {course}', 0, 1)
        self.ln(10)

    def fee_table(self, header, data):
        # Colors, line width and bold font
        self.set_fill_color(200, 220, 255)
        self.set_text_color(0)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(.3)
        self.set_font('', 'B')
        # Header
        w = [130, 40]
        for i in range(len(header)):
            self.cell(w[i], 7, header[i], 1, 0, 'C', 1)
        self.ln()
        # Color and font restoration
        self.set_fill_color(224, 235, 255)
        self.set_text_color(0)
        self.set_font('')
        # Data
        fill = 0
        for row in data:
            self.cell(w[0], 6, row[0], 'LR', 0, 'L', fill)
            self.cell(w[1], 6, f'{row[1]:,.2f}', 'LR', 0, 'R', fill)
            self.ln()
            fill = not fill
        self.cell(sum(w), 0, '', 'T')
        self.ln()

    def fee_summary(self, total_fees, total_paid, balance):
        self.set_font('Arial', 'B', 12)
        self.cell(130, 10, 'Total Fees:', 0, 0, 'R')
        self.cell(40, 10, f'{total_fees:,.2f}', 0, 1, 'R')
        self.cell(130, 10, 'Total Paid:', 0, 0, 'R')
        self.cell(40, 10, f'{total_paid:,.2f}', 0, 1, 'R')
        self.cell(130, 10, 'Balance:', 0, 0, 'R')
        self.set_font('Arial', 'B', 14)
        self.cell(40, 10, f'{balance:,.2f}', 0, 1, 'R')


# Instantiation of inherited class
pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.set_font('Times', '', 12)

# Student Details
pdf.student_details('John Doe', 'S/01/2024', 'Computer Science')

# Fee Table
table_header = ['Description', 'Amount (KES)']
fee_data = [
    ['Tuition Fees', 50000],
    ['Registration Fees', 2000],
    ['Library Fees', 1000],
    ['Examination Fees', 1500],
]
pdf.fee_table(table_header, fee_data)

# Fee Summary
total_fees = sum(item[1] for item in fee_data)
total_paid = 25000
balance = total_fees - total_paid
pdf.fee_summary(total_fees, total_paid, balance)

pdf.output('fee_statement.pdf')