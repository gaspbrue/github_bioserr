import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from sklearn.cluster import KMeans
import random
from matplotlib.colors import ListedColormap


# Load the hyperspectral image from the .bil file
def charger_imageHS(path_to_bil,path_to_hdr):
    data = np.fromfile(path_to_bil, dtype=np.float32)  # Read binary data, WARNING : dtype=np.float32 important !
    data = data.reshape((-1, 1))  # Reshape the data into a single column   
    # Load the metadata from the .hdr file
    with open(path_to_hdr, 'r') as hdr_file:
        for line in hdr_file:
            if line.startswith('samples'):
                samples = int(line.split('=')[1])
            elif line.startswith('lines'):
                lines = int(line.split('=')[1])
            elif line.startswith('bands'):
                bands = int(line.split('=')[1])
                break
        data = data.reshape((lines, bands, samples))  # Reshape the data with the BIl interleave (important !)
        image = np.transpose(data, (0,2,1))  # Permute the axes to match the shape of the image
        pdb.set_trace()
        image = image[40:400, 190:620, :210]  # We keep only the parts of the image and the portion of the spectrum that matter to us (380nm - 820nm)
        image = np.nan_to_num(image)

        min_value_image = np.min(image)
        max_value_image = np.max(image)
        print("Max value of the image :", min_value_image)
        print("Min value of the image :", max_value_image)
        
    return image






#2 Visualize a selected band
def afficher_bande(image, n, c, plante):
    os.makedirs(f"Images/Plante_{plante}/Bande", exist_ok=True)
    filename = f'Images/Plante_{plante}/Bande/Image {c}, Bande {n}'
    bande = image[:, :, n]

    plt.clf()
    plt.imshow(bande, cmap='gray')
    plt.title(f'Image {c}, Bande {n}')
    plt.savefig(filename)
    plt.draw()
    
    """
    # Min and max value of the band
    min_value_bande = np.min(bande)
    max_value_bande = np.max(bande)
    print("Valeur minimale de la bande :", min_value_bande)
    print("Valeur maximale de la bande :", max_value_bande)
    """




# Visualize the spectrum of a specific pixel
def afficher_spectre_pixel(image, pixel_position, c, plante): # eE: pixel_position = (250, 300)
    os.makedirs(f"Images/Plante_{plante}/Spectre_pixel", exist_ok=True)
    filename = f'Images/Plante_{plante}/Spectre_pixel/Image {c}, Pixel {pixel_position}'
    spectre_pixel = image[pixel_position[0], pixel_position[1], :]
    plt.clf()
    plt.plot(spectre_pixel)
    plt.xlabel('Bande')
    plt.ylabel('Réflectance')
    plt.title(f'Image {c}, Spectre de réflectance du pixel {pixel_position}')
    plt.savefig(filename)
    plt.draw()
    




#5 Visualize the image using a combination of bands
def afficher_bandes(image, selected_bands, c, plante): # ex: selected_bands = [150,200,230]
    os.makedirs(f"Images/Plante_{plante}/Bandes", exist_ok=True)    
    filename = f'Images/Plante_{plante}/Bandes/Image {c}, Bandes {selected_bands}'
    selected_pixels = image[:, :, selected_bands]
    plt.clf()
    plt.imshow(selected_pixels)
    plt.title(f'Image {c}, Bandes {selected_bands}')
    plt.colorbar()
    plt.savefig(filename) 
    plt.draw()













# Visualize Vegetation Indices
def afficher_IV(image, image_not_normalized, c, plante, IV):
    
    liste_IV = []
    
    for x in IV:
        if x == 'NDVI': 
            epsilon = 0.0000000001  # We introduce epsilon only to make sure we don't divide by 0
            os.makedirs(f"Images/Plante_{plante}/indice_ndvi", exist_ok=True)
            red_band = image[:, :, 105]    # Red band
            nir_band = image[:, :, 200]
            ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
            filename = f'Images/Plante_{plante}/indice_ndvi/Image {c}, {x}'
            plt.clf()
            plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(label='NDVI')
            plt.title(f'Image {c}, NDVI')
            plt.savefig(filename)
            plt.draw()
            liste_IV.append(ndvi)
        
        if x == 'NIR':
            os.makedirs(f"Images/Plante_{plante}/indice_nir", exist_ok=True)
            filename = f'Images/Plante_{plante}/indice_nir/Image {c}, nir'
            nir_band = image[:, :, 200]
            plt.clf()
            plt.imshow(nir_band, cmap='RdYlGn', vmin = -0.5, vmax = 1)
            plt.colorbar(label='nir')
            plt.title(f'Image {c}, nir')
            plt.savefig(filename)
            plt.draw()
            liste_IV.append(nir_band)

        if x == 'SR4':
            epsilon = 0.000000000000000001
            os.makedirs(f"Images/Plante_{plante}/indice_sr4", exist_ok=True)
            filename = f'Images/Plante_{plante}/indice_sr4/Image {c}, sr4'
            r700 = image[:, :, 153]
            r670 = image[:, :, 138]  
            sr4 = r700 / (r670 + epsilon)
            plt.clf()
            plt.imshow(sr4, cmap='RdYlGn', vmin=-10, vmax=10)
            plt.colorbar(label='sr4')
            plt.title(f'Image {c}, sr4')
            plt.savefig(filename)
            plt.draw()
            liste_IV.append(sr4)

        if x == 'mRENDVI':      
            os.makedirs(f"Images/Plante_{plante}/indice_mRENDVI", exist_ok=True)
            epsilon = 0.00000000001
            filename = f'Images/Plante_{plante}/indice_mRENDVI/Image {c}, mRENDVI'
            r705 = image[:, :, 155]   
            r750 = image[:, :, 177]  
            r445 = image[:,:, 31]
            mRENDVI = (r750 - r705) / (r750 + r705 - 2*r445 + epsilon)
            plt.clf()
            plt.imshow(mRENDVI, cmap='RdYlGn', vmin = -1, vmax = 1)
            plt.colorbar(label='mRENDVI')
            plt.title(f'Image {c}, mRENDVI')
            plt.savefig(filename)
            plt.draw()
            liste_IV.append(mRENDVI)

        if x == 'PSRI':
            epsilon = 0.00000000001
            os.makedirs(f"Images/Plante_{plante}/indice_PSRI", exist_ok=True)
            filename = f'Images/Plante_{plante}/indice_PSRI/Image {c}, PSRI'
            r500 = image[:, :, 55]
            r680 = image[:, :, 143]  
            r750 = image[:,:, 178]
            PSRI = (r750 - r500) / (r750 + epsilon)
            plt.clf()
            plt.imshow(PSRI, cmap='RdYlGn', vmin = -1.5, vmax = 1.5)
            plt.colorbar(label='PSRI')
            plt.title(f'Image {c}, PSRI')
            plt.savefig(filename)
            plt.draw()
            liste_IV.append(PSRI)

        if x == 'Crit_order':   
            os.makedirs(f"Images/Plante_{plante}/indice_Crit_order", exist_ok=True)
            epsilon = 0.0000000001
            filename = f'Images/Plante_{plante}/indice_Crit_order/Image {c}, Crit_order'
            r500 = image[:, :, 55]
            r680 = image[:, :, 143]  
            r750 = image[:,:, 178]
            PSRI = (r680 - r500) / r750
            r705 = image[:, :, 155]
            r445 = image[:,:, 31]
            mRENDVI = (r750 - r705) / (r750 + r705 - 2*r445 + epsilon)
            Crit_order = mRENDVI - PSRI
            plt.clf()
            plt.imshow(Crit_order, cmap='RdYlGn', vmin = -1.2, vmax = 1.2)
            plt.colorbar(label='Crit_order')
            plt.title(f'Image {c}, Crit_order')
            plt.savefig(filename)
            plt.draw()
            liste_IV.append(Crit_order)
    
    return liste_IV





















"""
# If we want to extract of region of interest
start_line = 100 
end_line = 200
start_sample = 50 
end_sample = 150  
start_band = 10
end_band = 220 
roi = image[start_line:end_line, start_sample:end_sample, start_band:end_band]
"""












# To remove the background from the image of the plant, using K-Means and then election only the clusters with NDVI > ndvi_threshold and NIR > nir_threshold
def background_elimination(image, plante, c, ndvi_threshold=0.3, nir_threhold = 0.4, n_clusters=10):
    os.makedirs(f"Images/Plante_{plante}/Image_Clusters", exist_ok=True)
    os.makedirs(f"Images/Plante_{plante}/Spectre_Centroids", exist_ok=True)
    
    #K-Means clustering to separate plant and background pixels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    flattened_image = image.reshape((-1, image.shape[-1]))
    kmeans.fit(flattened_image)

    # Get the cluster centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
            
        
    # We display the image with pixels grouped according to their clusters
    cluster_colors = ['#'+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(n_clusters)]  
    clustered_image = labels.reshape(image.shape[:-1])
    plt.clf()
    plt.imshow(clustered_image, cmap=ListedColormap(cluster_colors))
    plt.title(f'Pixels regroupés par clusters ({n_clusters})')
    filename = f'Images/Plante_{plante}/Image_Clusters/Image_{c}, Clusters_{n_clusters}'
    plt.savefig(filename)
    plt.draw()
    




    # Select plant clusters based on NDVI and NIR thresholds
    epsilon = 0.00000001
    red_values = centroids[:, 138]
    nir_values = centroids[:, 200]
    ndvi_values = (nir_values - red_values) / (nir_values + red_values + epsilon)
    plant_indices = np.logical_and(ndvi_values > ndvi_threshold, nir_values > nir_threhold)

    # We display the spectrum of the centroids
    plt.clf()
    for i in range(centroids.shape[0]):
        if plant_indices[i]:        
            plt.plot(centroids[i], label=f'Cluster {i + 1}', color=cluster_colors[i])
    plt.ylim(0, 1.2)
    plt.xlabel('Ordonnée (red = 138, nir = 200)')
    plt.ylabel('Valeurs')
    plt.title(f'Spectre des Centroids, {n_clusters} clusters ')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Adjust the spacing so that the caption does not overlap with the image
    plt.grid(True)
    os.makedirs(f"Images_Clustering", exist_ok=True)
    filename = f'Images/Plante_{plante}/Spectre_Centroids/Image_{c}, Clusters = {n_clusters}'
    plt.savefig(filename)
    plt.draw()
    

    # Create a copy of the original image to insert the plant pixels and fill the rest with zeros
    image_with_plant = np.zeros_like(image)
    selected_cluster_indices = np.where(plant_indices)[0] 
    labels = labels.reshape((image.shape[0], image.shape[1]))
    mask = np.isin(labels, selected_cluster_indices)
    plant_pixels = np.copy(image)
    # Set to zero the pixels that are not included in the mask
    plant_pixels[~mask] = 0

    return plant_pixels
    




def smooth_image(image):
        image[image > 1] = 1  
        image[image < 0] = 0  
        return image


def normalisation_minmax(image):
        # Normalize the values between 0 and 1
        min_value = np.min(image)
        max_value = np.max(image)
        image = (image - min_value) / (max_value - min_value)
        return image
    

def normalisation_L2(image):
    # Spectral normalization for each pixel, L2 Euclidean norm normalization
    image = np.apply_along_axis(lambda x: x / np.linalg.norm(x), axis=2, arr=image)
    return image






