from scanpy import read_h5ad


class DataHandler:

    def __init__(self, file_location):
        self.file_location = file_location
        self.annotation_data = self.read_data()

    def read_data(self):
        annotation_data = read_h5ad(filename=self.file_location)

        return annotation_data

    def print_data(self):
        print(self.annotation_data.X)


if __name__ == '__main__':
    file = "/home/ubuntu/Downloads/hca_heart_immune_download.h5ad"
    dh = DataHandler(file)
    dh.print_data()
