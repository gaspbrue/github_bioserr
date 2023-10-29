import os
import numpy as np
import matplotlib.pyplot as plt
from utils_plantes import afficher_bande
from utils_plantes import afficher_bandes
from utils_plantes import afficher_spectre_pixel
from utils_plantes import charger_imageHS
from utils_plantes import smooth_image
from utils_plantes import normalisation_L2
from utils_plantes import normalisation_minmax
from utils_plantes import afficher_IV
import pdb
from utils_plantes import background_elimination 
from sklearn.preprocessing import normalize


# Hyperparameters 
plante = 5 # The plant we want to visualize
selected_band = 205
selected_pixel = (176, 514)
IV = ['NDVI', 'SR4', 'mRENDVI','PSRI','Crit_order','NIR']
bands = [50, 100, 200]    # We can choose these bands to have a better visibility
bands_RGB = [21, 76, 91]   # We can chosse these bands if we want to visualize RGB
bands_CIR = [190,200,205]
folder_path = f'Data Plantes/Plante {plante}'
file_list = os.listdir(folder_path) # Get the list of files in the folder
file_list.sort()
os.makedirs("Images", exist_ok=True)
os.makedirs(f"Images/Plante_{plante}", exist_ok=True)





# Load the hyperspectral images
path_to_hdr = {}
c=0
for file_name in file_list:
    if file_name.endswith('.hdr'):
        
        # Load the hyperspectral images
        c+=1
        path_to_hdr[f'path_to_hdr_{c}'] = os.path.join(folder_path, file_name)
        file_name_bil = file_name.replace('.bil.hdr', '')
        file_name_bil += '.bil'
        path_to_bil = os.path.join(folder_path, file_name_bil)
        image = charger_imageHS(path_to_bil, path_to_hdr[f'path_to_hdr_{c}'])
        image = smooth_image(image)  # We ensure that outlier values greater than 1 are set to 1, and negative outlier values are set to 0.
        
        """
        # Choose the normalization
        image = normalisation_L2(image)    
        image = normalisation_minmax(image) 
                         
        # If you want to remove the plants without the background (not necessary)
        image = background_elimination(image, plante, c)   # si on veut éliminer le background
        
        print("Shape de l'image :", image.shape)
        print("Nombre de bandes :", image.shape[2])
        print("Nombre de pixels :", image.shape[0] * image.shape[1])
        """
        
        # Visualize Bands, Vegetations Indices maps, Spectrum of a choosen pixel
        afficher_bande(image, selected_band, c, plante)
        afficher_spectre_pixel(image, selected_pixel, c, plante)
        afficher_bandes(image, bands, c, plante)
        liste_IV = afficher_IV(image, image, c, plante, IV)

        
        
        
