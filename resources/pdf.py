# == Standard ==
from datetime import datetime

# == Third-party ==
from fpdf import FPDF

# Constants
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
    Builds a report pdf document.
    """

    def __init__(self):
        """Constructor
        """
        self.pdf = FPDF(format="A4")

        # Start positions
        self.x: int = 0
        self.y: int = 0

    def set_font_heading(self):
        """
        Sets the font for a heading.
        """
        self.pdf.set_font(family="helvetica", style=STYLE_HEADING, size=SIZE_HEADING)

    def set_font_section(self):
        """
        Sets the font for a section heading.
        """
        self.pdf.set_font(family="helvetica", style=STYLE_HEADING, size=SIZE_SECTION)

    def set_font_text(self):
        """
        Sets the font for the text.
        """
        self.pdf.set_font(family="helvetica", size=SIZE_TEXT)

    def add_page(self):
        """
        Adds a page to the document.
        """
        self.pdf.add_page()

    def add_image(self, title: str, image_path: str):
        """
        Adds an image to the document.

        :param title: Heading for the image
        :param image_path: Path to the image
        """
        self.add_linebreak(10)
        self.set_font_section()
        self.pdf.write(txt=title)
        self.pdf.ln()
        self.add_horizontal_line()

        self.pdf.ln(2)
        self.pdf.image(image_path)

    def add_horizontal_line(self):
        """
        Adds a horizontal line to the document.
        """
        # Add space on the top
        self.pdf.ln(2)
        page_width = self.pdf.w
        # Calculate line length
        line_length = (LINE_LENGTH / 100) * page_width

        # Get position on page
        x_start = (page_width - line_length) / 2
        x_end = x_start + line_length
        y = self.pdf.get_y()

        # Draw line
        self.pdf.line(x_start, y, x_end, y)

    def add_linebreak(self, spacing: float):
        """
        Adds a linebreak to the document.

        :param spacing: Spacing to the block above
        """
        self.y += spacing
        self.pdf.ln(self.y)

    def insert_header(self, header_image: str):
        """
        Inserts the header block.

        :param header_image: Letter head image
        """
        self.pdf.image(header_image, self.x, self.y, self.pdf.w)

        self.add_linebreak(0)
        self.set_font_text()
        self.pdf.write(txt=f"{datetime.now().strftime('%d.%m.%Y')}")

        self.pdf.ln(20)
        self.set_font_heading()
        self.pdf.write(txt=f"Statistical Report")

    def insert_metadata(self, filename: str, num_samples: int, num_genes: int):
        """
        Inserts the metadata block.

        :param filename: Name of the data file
        :param num_samples: Number of samples in data
        :param num_genes: Number of genes in data
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

    def export_pdf(self, filename: str):
        """
        Exports the pdf.

        :param filename: filename for the pdf
        """
        self.pdf.output(filename)
