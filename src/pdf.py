from datetime import datetime
from os import getcwd

from fpdf import FPDF
from fpdf.enums import XPos, YPos

STYLE_HEADING = "B"
SIZE_HEADING = 24
SIZE_SECTION = 16
SIZE_TEXT = 11
LINE_LENGTH = 90

HEADER_IMAGE = "/home/ubuntu/Projects/Master-Internship/resources/pdf_header.png"

METADATA_START_X = 10
METADATA_FILENAME = "Filename:"
METADATA_SAMPLES = "#Samples:"
METADATA_GENES = "#Genes:"
METADATA_DONORS = "#Donors:"
METADATA_CELL_TYPES = "#Cell Types:"
METADATA_AVG_EXPRESSION = "Average:"
METADATA_TOP5_HIGHEST = "Top 5 Highest:"
METADATA_TOP5_LOWEST = "Top 5 Lowest:"
METADATA_0_GENES = "Non Expressed Genes:"
METADATA_NON_0_GENES = "Non 0 Genes:"

SECTION_DONORS = "Donors"
SECTION_CELL_TYPES = "Cell Types"
SECTION_EXPRESSION = "Expression"

DONORS = [["Donor", "Samples"], ["Donor1", "50"], ["Donor2", "1000"], ["Donor3", "30"]]
CELL_TYPES = [["Cell Type", "Samples"], ["Type1", "50"], ["Type2", "1000"], ["Type3", "30"]]
TOP5_HIGHEST = [["Expression", "Occurrence"], ["1", "50"], ["2", "1000"], ["3", "30"], ["4", "50000"], ["5", "4"]]
TOP5_LOWEST = [["Expression", "Occurrence"], ["1", "50"], ["2", "1000"], ["3", "30"], ["4", "50000"], ["5", "4"]]
ZERO_GENES = ["Gene1", "Gene2", "Gene3"]
NON_ZERO_GENES = ["Gene1", "Gene2", "Gene3"]


class PdfBuilder:
    """
    Docstring
    """

    def __init__(self):
        """
        Docstring
        """
        self.pdf = FPDF(format="A4")

        # Start positions
        self.x: int = 0
        self.y: int = 0

    def set_font_heading(self):
        """
        Docstring
        """
        self.pdf.set_font(family="helvetica", style=STYLE_HEADING, size=SIZE_HEADING)

    def set_font_section(self):
        """
        Docstring
        """
        self.pdf.set_font(family="helvetica", style=STYLE_HEADING, size=SIZE_SECTION)

    def set_font_text(self):
        """
        Docstring
        """
        self.pdf.set_font(family="helvetica", size=SIZE_TEXT)

    def add_page(self):
        """
        Docstring
        """
        self.pdf.add_page()

    def add_table(self, data: list[list[str]]):
        """
        Docstring
        """
        with self.pdf.table(borders_layout="MINIMAL") as table:
            for data_row in data:
                row = table.row()
                for entry in data_row:
                    row.cell(entry)

    def add_list(self, data: list[str]):
        """
        Docstring
        """
        self.pdf.ln()
        for item in data:
            self.pdf.cell(txt=item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def add_horizontal_line(self):
        """
        Docstring
        """
        self.pdf.ln(2)
        page_width = self.pdf.w
        line_length = (LINE_LENGTH / 100) * page_width

        x_start = (page_width - line_length) / 2
        x_end = x_start + line_length
        y = self.pdf.get_y()

        self.pdf.line(x_start, y, x_end, y)

    def add_linebreak(self, spacing: float):
        """
        Docstring
        """
        self.y += spacing
        self.pdf.ln(self.y)

    def insert_header(self, header_image: str):
        """
        Docstring
        """
        self.pdf.image(header_image, self.x, self.y, self.pdf.w)

        self.add_linebreak(0)
        self.set_font_text()
        self.pdf.write(txt=f"{datetime.now().strftime('%d.%m.%Y')}")

        self.pdf.ln(20)
        self.set_font_heading()
        self.pdf.write(txt=f"Statistical Report")

    def insert_metadata(self, filename: str, num_samples: int, num_genes: int, num_donors: int,
                        donors: list[list[str]], num_cell_types: int, cell_types: list[list[str]],
                        avg_expression: float, top5_highest: list[list[str]], top5_lowest: list[list[str]],
                        zero_genes, non_zero_genes):
        """
        Docstring
        """
        self.add_linebreak(55)
        self.set_font_heading()
        self.pdf.write(txt="Metadata")
        self.pdf.ln()
        self.add_horizontal_line()

        self.pdf.ln(10)
        self.set_font_text()

        # Filename
        self.pdf.set_x(METADATA_START_X)
        text = f"{METADATA_FILENAME} {filename}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="C")

        # Samples
        new_x = METADATA_START_X + self.pdf.get_string_width(text) + 5
        self.pdf.set_x(new_x)
        text = f"{METADATA_SAMPLES} {num_samples}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="C")

        # Genes
        new_x = new_x + self.pdf.get_string_width(text) + 5
        self.pdf.set_x(new_x)
        text = f"{METADATA_GENES} {num_genes}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="C")

        # Donors
        self.pdf.ln(15)
        self.set_font_section()
        self.pdf.write(txt=SECTION_DONORS)

        self.pdf.ln(7)
        self.set_font_text()
        text = f"{METADATA_DONORS} {num_donors}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")

        self.pdf.ln(7)
        self.add_table(donors)

        # Cell types
        self.pdf.ln(15)
        self.set_font_section()
        self.pdf.write(txt=SECTION_CELL_TYPES)

        self.pdf.ln(7)
        self.set_font_text()
        text = f"{METADATA_CELL_TYPES} {num_cell_types}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")

        self.pdf.ln(7)
        self.add_table(cell_types)

        # Expressions
        self.pdf.ln(15)
        self.set_font_section()
        self.pdf.write(txt=SECTION_EXPRESSION)

        self.pdf.ln(7)
        self.set_font_text()
        text = f"{METADATA_AVG_EXPRESSION} {avg_expression}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")

        self.pdf.ln(10)
        text = f"{METADATA_TOP5_HIGHEST}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")
        self.pdf.ln()
        self.add_table(top5_highest)

        self.pdf.ln(7)
        text = f"{METADATA_TOP5_LOWEST}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")
        self.pdf.ln()
        self.add_table(top5_lowest)

        # 0 Expression
        self.pdf.ln(10)
        text = f"{METADATA_0_GENES}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")
        self.add_list(zero_genes)

        # >0 Expression
        self.pdf.ln(7)
        text = f"{METADATA_NON_0_GENES}"
        self.pdf.cell(w=(self.pdf.get_string_width(text)), txt=text, align="L")
        self.add_list(non_zero_genes)

    def export_pdf(self, filename: str):
        """
        Docstring
        """
        self.pdf.output(filename)


if __name__ == '__main__':
    new = PdfBuilder()
    new.add_page()
    new.insert_header(HEADER_IMAGE)
    new.insert_metadata("test.h5ad", 500, 500, 14, donors=DONORS,
                        num_cell_types=20, cell_types=CELL_TYPES, avg_expression=13.5, top5_highest=TOP5_HIGHEST,
                        top5_lowest=TOP5_LOWEST, zero_genes=ZERO_GENES, non_zero_genes=NON_ZERO_GENES)
    print(f"{getcwd()}/output.pdf")
    new.export_pdf(f"{getcwd()}/output.pdf")
