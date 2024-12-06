import numpy as np

class MaterialData:

    def __init__(self, data):
        self.data = data


class LUT:

    # Parametrize each medium block
    # "Pack" it all in the output (Full image and metadata)
    # Unpack function is a form of init

    def __init__(self, settings_object):
        self._settings = settings_object.GetSettings()
        settings = self._settings 

        image_size = settings['column_count'] + len(settings_object.getAux())
        image_size_x = image_size
        self.density_block_size = int(np.floor((image_size) / settings['med_ior_rows']))

        if 'rows' in settings:
            self.density_block_size = settings['rows'] # Row override
            image_size_x = self.density_block_size

        self._image = np.zeros((image_size_x, image_size, 3), dtype=np.float32)

        return

    def modifyLUT(self, block_index, row_number, row):
        self.image = self._image[0]

        return
    
    def modifyLUT(self, block_index, block):
        self._image

        return
    
    def modifyLUT(self, column):
        self._image

        return     
    
    def getSettings(self):
        return self._settings

    def packLUT(self, ):

        pass

    def unpackLUT(self, ):

        pass

    def saveLUTAsIamge(self, ):

        pass
    
    def readLUTFromImage(self, ):

        pass

class Settings:

    def __init__(self, main):
        self.__ActivatedMode = None
        self.__Settings = main

    def getSettings(self):
        return self.__Settings
    
    def getAux(self):
        return np.array([item.strip() for item in self.__Settings['aux_columns'].split(',')])