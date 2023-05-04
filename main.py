import data_handler as dh


def main():
    file = '/media/sf_Share/global_raw.h5ad'
    handler = dh.DataHandler(file)
    # handler.show_data()
    handler.tpm()
    # handler.create_input_vector(cell_type='Fibroblast')
    # control_data, training_data = handler.separate_individual()


if __name__ == '__main__':
    main()
