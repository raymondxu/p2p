
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateLatentDataLoader(opt):
    from data.latent_dataset_data_loader import LatentDatasetDataLoader
    data_loader = LatentDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
