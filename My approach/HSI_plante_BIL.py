import numpy as np
import matplotlib.pyplot as plt

# Chemin vers les fichiers .hdr et .bin
path_to_hdr = 'Data Plantes/Plante 1/2023-07-12T13-35-52_915+0200_#PLANTE_1#.bil.hdr'
path_to_bil = 'Data Plantes/Plante 1/2023-07-12T13-35-52_915+0200_#PLANTE_1#.bil'

# Charger l'image hyperspectrale à partir du fichier .bin
data = np.fromfile(path_to_bil, dtype=np.float32)  # Lire les données binaires  # attention dtype=np.float32 important !
data = data.reshape((-1, 1))  # Remodeler les données en une colonne

# Charger les métadonnées à partir du fichier .hdr
with open(path_to_hdr, 'r') as hdr_file:
    for line in hdr_file:
        if line.startswith('samples'):
            samples = int(line.split('=')[1])
        elif line.startswith('lines'):
            lines = int(line.split('=')[1])
        elif line.startswith('bands'):
            bands = int(line.split('=')[1])
            break


data = data.reshape((lines, bands, samples))  # Remodeler les données avec l'interleave bip
# Permuter les axes pour correspondre à la forme de l'image
image = np.transpose(data, (0, 2, 1))
image = image[50:400, 190:620   , :]
image = np.nan_to_num(image)


"""
# Normaliser les valeurs de l'image entre 0 et 1
min_value = np.min(image)
max_value = np.max(image)
image = (image - min_value) / (max_value - min_value)
"""





# Trouver la valeur minimale et maximale de l'image
min_value_image = np.min(image)
max_value_image = np.max(image)
print("Valeur minimale de l'image :", min_value_image)
print("Valeur maximale de l'image :", max_value_image)




#1 Afficher les informations de l'image
print("Shape de l'image :", image.shape)
print("Nombre de bandes :", image.shape[2])
print("Nombre de pixels :", image.shape[0] * image.shape[1])


#2 Afficher une bande spécifique
band_index = 110   # Indice de la bande à afficher
bande = image[:, :, band_index]
"""
plt.imshow(bande, cmap='gray', vmin=-1, vmax=1)
"""
plt.imshow(bande, cmap='gray')
plt.title(f'Bande {band_index}')
plt.show()

# Trouver la valeur minimale et maximale de la bande
min_value_bande = np.min(bande)
max_value_bande = np.max(bande)
print("Valeur minimale de la bande :", min_value_bande)
print("Valeur maximale de la bande :", max_value_bande)




#3 Afficher le spectre d'un pixel spécifique
pixel_position = (200, 400)  # Position du pixel dans l'image
spectre_pixel = image[pixel_position[0], pixel_position[1], :]
plt.plot(spectre_pixel)
plt.xlabel('Bande')
plt.ylabel('Réflectance')
plt.title(f'Spectre de réflectance du pixel {pixel_position}')
plt.show()




"""
#4 Créer une composition colorée en fausses couleurs
red_band = image[:, :, 10]  # Indice de la bande rouge
green_band = image[:, :, 20]  # Indice de la bande verte
blue_band = image[:, :, 30]  # Indice de la bande bleue
# Normaliser les bandes pour obtenir des valeurs entre 0 et 1
red_band_normalized = red_band / np.max(red_band)
green_band_normalized = green_band / np.max(green_band)
blue_band_normalized = blue_band / np.max(blue_band)
# Créer l'image en fausses couleurs
rgb_image = np.stack((red_band_normalized, green_band_normalized, blue_band_normalized), axis=2)
# Afficher l'image en fausses couleurs
plt.imshow(rgb_image)
plt.axis('off')
plt.title('Composition en fausses couleurs')
plt.show()
"""





#5 Afficher l'image complète en utilisant une combinaison de bandes
selected_bands = [50,150,250]  # Sélectionnez les bandes que vous souhaitez afficher
# Extrait les valeurs des pixels pour les bandes sélectionnées
selected_pixels = image[:, :, selected_bands]
# Affichez l'image en utilisant matplotlib
plt.imshow(selected_pixels)
plt.title(f'Images Hyperspectrale, Bande {selected_bands}')
plt.colorbar()
plt.show()




# Spécifier les coordonnées de la région d'intérêt
start_line = 100  # Ligne de départ
end_line = 200    # Ligne de fin (non inclusive)
start_sample = 50  # Colonne de départ
end_sample = 150   # Colonne de fin (non inclusive)
start_band = 10   # Bande de départ
end_band = 20     # Bande de fin (non inclusive)

# Extraire la région d'intérêt de l'image
roi = image[start_line:end_line, start_sample:end_sample, start_band:end_band]






















# Extraire les bandes rouges et proches infrarouges (NIR)
red_band = image[:, :, 133]  # Remplacez 29 par l'indice de la bande rouge dans votre image
nir_band = image[:, :, 211]  # Remplacez 45 par l'indice de la bande NIR dans votre image

# Calculer l'indice NDVI
ndvi = (nir_band - red_band) / (nir_band + red_band)

# Calculer la valeur maximale, minimale et les coordonnées (indices de lignes et de colonnes) des pixels correspondants pour red_band
red_max_value = np.max(red_band)
red_min_value = np.min(red_band)
red_max_coords = np.unravel_index(np.argmax(red_band), red_band.shape)
red_min_coords = np.unravel_index(np.argmin(red_band), red_band.shape)

# Calculer la valeur maximale, minimale et les coordonnées (indices de lignes et de colonnes) des pixels correspondants pour nir_band
nir_max_value = np.max(nir_band)
nir_min_value = np.min(nir_band)
nir_max_coords = np.unravel_index(np.argmax(nir_band), nir_band.shape)
nir_min_coords = np.unravel_index(np.argmin(nir_band), nir_band.shape)

# Calculer la valeur maximale, minimale et les coordonnées (indices de lignes et de colonnes) des pixels correspondants pour ndvi
ndvi_max_value = np.max(ndvi)
ndvi_min_value = np.min(ndvi)
ndvi_max_coords = np.unravel_index(np.argmax(ndvi), ndvi.shape)
ndvi_min_coords = np.unravel_index(np.argmin(ndvi), ndvi.shape)

# Afficher la carte NDVI
plt.imshow(ndvi, cmap='RdYlGn')  # Utilisez le colormap RdYlGn pour afficher en faux couleurs (rouge-jaune-vert)
plt.colorbar(label='NDVI')
plt.title('Carte NDVI')
plt.show()

# Afficher les résultats
print("Red Band - Maximum Value:", red_max_value)
print("Red Band - Minimum Value:", red_min_value)
print("Red Band - Coordinates of Maximum Value (row, col):", red_max_coords)
print("Red Band - Coordinates of Minimum Value (row, col):", red_min_coords)

print("NIR Band - Maximum Value:", nir_max_value)
print("NIR Band - Minimum Value:", nir_min_value)
print("NIR Band - Coordinates of Maximum Value (row, col):", nir_max_coords)
print("NIR Band - Coordinates of Minimum Value (row, col):", nir_min_coords)

print("NDVI - Maximum Value:", ndvi_max_value)
print("NDVI - Minimum Value:", ndvi_min_value)
print("NDVI - Coordinates of Maximum Value (row, col):", ndvi_max_coords)
print("NDVI - Coordinates of Minimum Value (row, col):", ndvi_min_coords)