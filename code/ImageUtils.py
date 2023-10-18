import numpy as np
from matplotlib import pyplot as plt
import torchvision

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))
    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    image = preprocess_image(image, training) # If any.
    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])
    ### END CODE HERE
    return image

def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        # Resize the image to add four extra pixels on each side.        
        image = np.pad(image, ((4,4), (4,4), (0,0)), constant_values=0)
        # Randomly crop a [32, 32] section of the image.
        h_init = np.random.randint(low=0, high=8)
        w_init = np.random.randint(low=0, high=8)
        image = image[h_init:h_init+32, w_init:w_init+32, :]
        # Randomly flip the image horizontally.
        if np.random.normal()>0 :
            image = np.fliplr(image)
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image)
    sd = np.sqrt(np.var(image)+1/(32*32*3)) # Take eps = 1/N
    image = (image - mean)/sd
    ### END CODE HERE
    return image



def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # Reshape the image
    image = image.reshape((3, 32, 32))
    # Transpose the image
    image = image.transpose(1,2,0)
    image = image.astype(int)
    #image = torchvision.utils.make_grid(image)
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE