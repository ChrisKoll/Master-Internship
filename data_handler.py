from scanpy import read_h5ad


class DataHandler:

    def __init__(self, file_location):
        self.file_location = file_location
        self.annotated_data = self.read_data()

    def read_data(self):
        annotated_data = read_h5ad(filename=self.file_location)

        return annotated_data

    def create_input_vector(self, cell_type: str = None, heart_region: str = None):
        ad = self.annotated_data

        if cell_type is not None:
            ad_subset = ad[ad.obs.cell_type == cell_type].X

            return ad_subset

        elif heart_region is not None:
            ad_subset = ad[ad.obs.region == heart_region].X

            return ad_subset

        else:
            print('Please provide a method argument...')
