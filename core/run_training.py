import core.fetch_300w_data as fd
import core.generate_params as gp
import core.train as tr

if __name__ == "__main__":
    fd.import_300lw_data()
    gp.generate_param_files()
    tr.run_training()