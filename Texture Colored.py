import cv2
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


image_path = 'RX-0.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


pixel_values = image_hsv.reshape((-1, 3)).astype(float)

k = 4

np.random.seed(0)
centroids = pixel_values[np.random.choice(pixel_values.shape[0], k, replace=False)]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(pixel_values, centroids):
    distances = np.linalg.norm(pixel_values[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(pixel_values, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = pixel_values[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(np.random.rand(3) * 255)
    return np.array(new_centroids)

max_iterations = 100
for _ in range(max_iterations):
    labels = assign_clusters(pixel_values, centroids)
    new_centroids = update_centroids(pixel_values, labels, k)
    if np.allclose(centroids, new_centroids, atol=1e-4):
        break
    centroids = new_centroids


segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)


custom_colors = [
    [255, 255, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
]


colored_segmented_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
for i in range(k):
    colored_segmented_image[labels.reshape(image.shape[0], image.shape[1]) == i] = custom_colors[i]


labels = ["Soft", "Rough", "Pattern", "Shadows"]
patches = [mpatches.Patch(color=np.array(color) / 255, label=label) for color, label in zip(custom_colors, labels)]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title('Segmented')
plt.imshow(colored_segmented_image)
plt.axis('off')
plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(-0.15, -0.1), ncol=1, fontsize=10)
plt.tight_layout()
plt.show()
