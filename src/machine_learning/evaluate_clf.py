from src.data_management import load_pkl_file


def load_test_evaluation_model_3(version):
    return load_pkl_file(f'outputs/{version}/evaluation_model_3.pkl')
