from pathlib import Path

import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import filters, measure, morphology
import SimpleITK as sitk

# Set image and output paths
OUTPUT_PATH = ... # Path("./output/ensemble_output2.npy")  # Save
INPUT_PATH = ... # Path("./input/10000/10000_1000000_adc.mha")


PATIENCE_BUFFER = 30

# Load output
with open(OUTPUT_PATH, mode="rb") as f:
    ensemble_output = np.load(f)

# Create 3d sphere
structuring_element = generate_binary_structure(3, 1)


def postprocessing():
    """
    Performs opening, contour identification, convex hull and closing on the ensemble_output.
    Additionally filters on the approximate are of the prostate based on strong assumptions
    """

    opened_ensemble_output_mask = binary_opening(
        ensemble_output, structure=structuring_element
    )
    # print(eroded_ensemble_output)
    labeled_image = label(opened_ensemble_output_mask)
    contours = regionprops(labeled_image)
    for contour in contours:
        hull = contour.convex_image
        opened_ensemble_output_mask[hull] = 1
    filled_image = binary_closing(opened_ensemble_output_mask, structure=structuring_element)

    processed_ensemble_output = np.zeros(ensemble_output.shape)
    processed_ensemble_output[filled_image] = ensemble_output[filled_image]
    approximate_prostate_bbox = locate_prostate_adc_single_side()
    minr, minc, maxr, maxc = approximate_prostate_bbox
    final_ensemble_output = np.zeros(processed_ensemble_output.shape)
    final_ensemble_output[
        :, minr + PATIENCE_BUFFER : maxr + PATIENCE_BUFFER, minc : maxc
    ] = processed_ensemble_output[
        :, minr + PATIENCE_BUFFER : maxr + PATIENCE_BUFFER, minc : maxc
    ]
    return final_ensemble_output


def locate_prostate_adc_single_side():
    """
    Find the prostate area with strong assumptions from a single side.
    #TODO: Create a 3d bounding box
    """
    # Load and preprocess the image
    image = sitk.ReadImage(INPUT_PATH)

    # image_dims = np.shape(image_array)
    output_dims = np.shape(ensemble_output)[::-1]
    print(output_dims)

    resampled_image = sitk.Resample(
        image,
        output_dims,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        (
            image.GetSpacing()[0] * image.GetWidth() / output_dims[0],
            image.GetSpacing()[1] * image.GetHeight() / output_dims[1],
            image.GetSpacing()[2] * image.GetDepth() / output_dims[2],
        ),
        image.GetDirection(),
        0.0,
        image.GetPixelIDValue(),
    )
    image_array = sitk.GetArrayFromImage(resampled_image)

    image_dims = np.shape(image_array)
    print(image_dims)
    output_dims = np.shape(ensemble_output)

    # Select a slice for processing (e.g., the middle slice)
    slice_index = image_array.shape[0] // 2
    slice_image = image_array[slice_index]

    # slice_output_index = ensemble_output.shape[0] // 2
    # slice_output = ensemble_output[slice_output_index] # can be used to plot the corresponding slice on the output

    # Step 1: Pre-processing
    # Apply Gaussian filter to remove noise
    smoothed_image = filters.gaussian(slice_image, sigma=3.0)

    # Step 2: Thresholding
    # Use Otsu's method to create a binary image
    threshold_value = filters.threshold_otsu(smoothed_image)
    binary_image = smoothed_image > threshold_value

    # Step 3: Morphological Operations
    # Remove small objects and fill small holes
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=500)
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=500)

    # Step 4: Contour Identification
    # Label connected components
    labeled_image = measure.label(cleaned_image)
    regions = measure.regionprops(labeled_image)
    # plt.imshow(labeled_image)
    # plt.show()
    # Step 5: Region Properties
    # Analyze properties to identify the prostate
    # Assuming prostate is the largest region in the cleaned binary image

    center = np.array(image_array.shape[1:3]) / 2
    closest_region = min(
        regions, key=lambda r: np.linalg.norm(center - np.array(r.centroid)[1:])
    )
    # largest_region = max(regions, key=lambda r: r.area)

    # Get the bounding box of the largest region
    # minr, minc, maxr, maxc = largest_region.bbox
    minr, minc, maxr, maxc = closest_region.bbox
    # print(f"bbox: {closest_region.bbox}")
    # Visualize the result
    minr = minr + PATIENCE_BUFFER
    maxr = maxr + PATIENCE_BUFFER
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(slice_image, cmap="gray")  # Display the original slice
    ax[0].set_title("Original Slice")
    ax[0].axis("off")

    ax[1].imshow(slice_image, cmap="gray")  # Display the original slice
    rect = plt.Rectangle(
        (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=2
    )
    ax[1].add_patch(rect)
    ax[1].set_title("Detected Prostate Region")
    ax[1].axis("off")

    plt.show()
    for i, ensemble_output_slice in enumerate(ensemble_output):
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

        ax[0].imshow(ensemble_output_slice, cmap="gray")  # Display the original slice
        ax[0].set_title("Original Slice")
        ax[0].axis("off")

        ax[1].imshow(ensemble_output_slice, cmap="gray")  # Display the original slice
        rect = plt.Rectangle(
            (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=2
        )
        ax[1].add_patch(rect)
        ax[1].set_title("Detected Prostate Region")
        ax[1].axis("off")

        plt.show()

    return closest_region.bbox


def plot_slices(array_of_image):
    """
    Plot slices from "I don't know exactly which" direction.
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Define the initial slice index
    initial_slice_index = 0

    # Plot the initial slice
    current_slice = array_of_image[initial_slice_index]
    img = ax.imshow(current_slice, cmap="viridis", vmin=0, vmax=1)
    colorbar = plt.colorbar(img, ax=ax)
    colorbar.set_label("Likelihood")

    # Define the slider's position and size
    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])

    # Create the slider
    slider = Slider(
        slider_ax,
        "Slice",
        0,
        array_of_image.shape[0] - 1,
        valinit=initial_slice_index,
        valstep=1,
    )

    # Define the update function
    def update(val):
        slice_index = int(slider.val)
        img.set_data(array_of_image[slice_index])
        ax.set_title(f"Slice {slice_index}")
        plt.draw()

    # Link the slider to the update function
    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    final_ensemble_output = postprocessing()
    # locate_prostate_adc_single_side()
    # plot_slices()
    # plot_slices(ensemble_output)
    plot_slices(final_ensemble_output)
    pass
