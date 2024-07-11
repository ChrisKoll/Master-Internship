# Standard libs
import configparser
from os import path

# Third-party libs
from anndata import AnnData, read_h5ad, write_h5ad

# Local imports
from . import constants as C
from normalize.normalize import Normalizer
from subset.subset import subset_adata
from train import model
from train import training


class Handler:
    """
    Class for handling h5ad data.
    """

    def __init__(self, args):
        """Constructor

        :param args: List of arguments
        """
        self.args = args
        self.adata = self.__read_data()

        self.__get_command()

    def __read_data(self):
        """
        Uses the AnnData function read_h5ad to import the AnnData object.

        :return: AnnData object
        """
        if not path.exists(self.file_location):
            raise ValueError(C.VALUE_ERR_IFP)
        else:
            self.adata = read_h5ad(filename=self.args[C.ARG_ANNDATA])

    def __get_command(self):
        """
        Executes the command according to the passed on arguments.
        """
        command = self.args[C.ARG_COMMAND]

        if command == C.ARG_COMMAND_EXPECTED1:
            self.__exec_subset()
        elif command == C.ARG_COMMAND_EXPECTED2:
            self.__exec_normalize()
        elif command == C.ARG_COMMAND_EXPECTED3:
            self.__exec_train()
        else:
            raise ValueError(C.VALUE_ERR_UC)

    def __exec_subset(self):
        """
        Docstring
        """
        if self.args[C.ARG_ROWS] is not None:
            arg_rows = self.args[C.ARG_ROWS]
        else:
            arg_rows = self.adata.n_obs * 0.1
        if self.args[C.ARG_COLUMNS] is not None:
            arg_columns = self.args[C.ARG_COLUMNS]
        else:
            arg_columns = self.adata.n_vars * 0.1
        if self.args[C.ARG_NO_SHUFFLE] is not None:
            arg_no_shuffle = self.args[C.ARG_NO_SHUFFLE]
        else:
            arg_no_shuffle = False

        self.__subset_create_subset([arg_rows, arg_columns], no_shuffle=arg_no_shuffle)

        # Create new filename
        filename, extension = self.args[C.ARG_ANNDATA].rsplit(".", 1)
        new_filename = f"{filename}_subset_{arg_rows}x{arg_columns}.{extension}"
        # Write to h5ad file
        self.adata.write_h5ad(new_filename)

    def __subset_create_subset(self, subset_params: list[int], *, no_shuffle: bool):
        """
        Docstring
        """
        if not no_shuffle:
            self.adata = subset_adata(
                self.adata,
                number_rows=subset_params[0],
                number_cols=subset_params[1],
            )
        if no_shuffle:
            self.adata = subset_adata(
                self.adata,
                number_rows=subset_params[0],
                number_cols=subset_params[1],
                shuffle=False,
            )

    def __exec_normalize(self):
        """
        Docstring
        """
        if self.args[C.ARG_METHOD] is not None:
            arg_method = self.args[C.ARG_METHOD]
        else:
            raise ValueError(C.VALUE_ERR_MA)

        self.adata = self.__normalization_convert_data(arg_method, self.adata)

        # Create new filename
        filename, extension = self.args[C.ARG_ANNDATA].rsplit(".", 1)
        new_filename = f"{filename}_normalized.{extension}"
        # Write to h5ad file
        self.adata.write_h5ad(new_filename)

    def __normalization_convert_data(
        self, arg_normalization: str, adata: AnnData
    ) -> AnnData:
        normalizer = Normalizer(arg_normalization, adata)

        return normalizer.run_normalization()

    def __exec_train(self):
        """
        Executes the train command.
        """
        if self.args[C.ARG_MODEL] is not None:
            arg_model = self.args[C.ARG_MODEL]
            model = self.__train_build_model(arg_model)
        if self.args[C.ARG_HYPERPARAMS] is not None:
            arg_hyperparams = self.args[C.ARG_HYPERPARAMS]
            hyperparams = self.__train_extract_hyperparams(arg_hyperparams)
        if self.args[C.ARG_SUBSET] is not None:
            arg_subset = self.args[C.ARG_SUBSET]
            self.__subset_create_subset(arg_subset)
        if self.args[C.ARG_NORMALIZATION] is not None:
            arg_normalization = self.args[C.ARG_NORMALIZATION]
            self.adata = self.__normalization_convert_data(
                arg_normalization,
                self.adata,
            )

        self.__train_fit_model(model, hyperparams)

    def __train_build_model(
        self, config_file: str
    ) -> model.Autoencoder | model.VariationalAutoencoder:
        """
        Builds the model according to the parameters in the ini file.
        """
        if not path.exists(config_file):
            raise ValueError(C.VALUE_ERR_IFP)
        else:
            config = configparser.ConfigParser()
            config.read(config_file)

            # Extract all key values from section
            keys = dict(config.items(C.M_CONFIG_SECTION2))
            if (
                config.get(C.M_CONFIG_SECTION1, C.M_CONFIG_S1_KEY1)
                == C.M_CONFIG_S1_K1_EXPECTED1
            ):
                autoencoder = model.Autoencoder(
                    size_input_layer=keys[C.M_CONFIG_S2_KEY1],
                    size_layer_one=keys[C.M_CONFIG_S2_KEY2],
                    size_layer_two=keys[C.M_CONFIG_S2_KEY3],
                    size_layer_three=keys[C.M_CONFIG_S2_KEY4],
                    size_latent_space=keys[C.M_CONFIG_S2_KEY5],
                )
                return autoencoder
            elif (
                config.get(C.M_CONFIG_SECTION1, C.M_CONFIG_S1_KEY1)
                == C.M_CONFIG_S1_K1_EXPECTED2
            ):
                var_autoencoder = model.VariationalAutoencoder(
                    size_input_layer=keys[C.M_CONFIG_S2_KEY1],
                    size_layer_one=keys[C.M_CONFIG_S2_KEY2],
                    size_layer_two=keys[C.M_CONFIG_S2_KEY3],
                    size_layer_three=keys[C.M_CONFIG_S2_KEY4],
                    size_latent_space=keys[C.M_CONFIG_S2_KEY5],
                )
                return var_autoencoder
            else:
                raise ValueError(C.VALUE_ERR_IK)

    def __train_extract_hyperparams(self, config_file: str) -> dict:
        """
        Extracts the hyperparameters from a config file.
        """
        if not path.exists(config_file):
            raise ValueError(C.VALUE_ERR_IFP)
        else:
            config = configparser.ConfigParser()
            config.read(config_file)

            # Extract all key values from section
            keys = dict(config.items(C.H_CONFIG_SECTION))

            return keys

    def __train_fit_model(self, model, hyperparams):
        """
        Docstring
        """
        trainer = training.Trainer()
