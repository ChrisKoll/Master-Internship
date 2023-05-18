from pathlib import Path

from scanpy import read_h5ad
from anndata import AnnData


class DataHandler:

    def __init__(self, file_location):
        self.file_location = file_location
        self.adata = self.read_data()
        self.adata_subset = None

    def read_data(self):
        annotated_data = read_h5ad(filename=self.file_location)

        return annotated_data

    def create_input_vector(self, cell_type: str = None, heart_region: str = None):
        adata = self.adata

        if cell_type is not None:
            self.adata_subset = adata[adata.obs.cell_type == cell_type]

        elif heart_region is not None:
            self.adata_subset = adata[adata.obs.region == heart_region]

        else:
            print('Please provide a method argument...')

    def separate_individual(self):
        adata = self.adata_subset
        donors = set(adata.obs.donor)

        print(*donors, sep=', ')
        chosen_donor = input('Choose a donor: ')

        control_data = adata[adata.obs.donor == chosen_donor]
        training_data = adata[adata.obs.donor != chosen_donor]

        return control_data, training_data

    def normalization(self):
        pass

    def subset_adata(self):
        subset = self.adata[:5000, :5000]
        path_obj = Path('/media/sf_Share/sample_data.h5ad')
        subset.write_h5ad(path_obj)

    '''
    def get_gene_length(self, gene_name: str):
        Entrez.email = 'christian.kolland@stud.uni-frankfurt.de'

        request = Entrez.epost('gene', id=gene_name)
        result = Entrez.read(request)

        web_env = result["WebEnv"]
        query_key = result["QueryKey"]
        data = Entrez.esummary(db="gene", webenv=web_env, query_key=query_key)
        print(Entrez.read(data))
    '''

    '''
    def get_gene_ids(self):
        gene_names = self.annotated_data.var_names.tolist()

        mg = mygene.MyGeneInfo()
        out = mg.querymany(gene_names, scopes='symbol', fields='entrezgene', species='human')
        print(out)
    '''

    def show_data(self):
        print(self.annotated_data.obs_names[:10].tolist())
        # print(self.annotated_data.obs_names[-10:].tolist())
        print(self.annotated_data.var_names[:10].tolist())
        # print(self.annotated_data.var_names[-10:].tolist())
