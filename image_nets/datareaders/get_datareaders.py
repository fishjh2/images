from image_nets.datareaders.CamVid import CamVidReader


def get_datareader(dataset_name: str):

    if dataset_name == 'CamVid':
        return CamVidReader()

    else:
        raise NotImplementedError(f'No dataset reader implemented for {dataset_name}')



