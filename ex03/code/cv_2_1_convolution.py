import cv2
import numpy as np

<<<<<<< HEAD
def convolve2D(image, kernel, padding=0, strides=2):
=======
    """
        taken from
        https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
        and fixed the indexing for the output image
    """
    # Do convolution instead of Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    xKernLeft = xKernShape // 2
    # since slicing end is exclusive, uneven kernel shapes would be too small
    xKernRight = int(np.around(xKernShape / 2.))
    yKernShape = kernel.shape[1]
    yKernUp = yKernShape // 2
    yKernDown = int(np.around(yKernShape / 2.))
    xImgShape = image.shape[1]
    yImgShape = image.shape[0]

    # Shape of Output Convolution
    # START TODO ###################
    xOutput = 1 + (xImgShape - xKernShape + 2 * padding) // strides
    yOutput = 1 + (yImgShape - yKernShape + 2 * padding) // strides
    # END TODO ###################
    output = np.zeros((yOutput, xOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        # START TODO ###################
        imagePadded = np.pad(image, (padding, padding))
        # END TODO ###################
    else:
        imagePadded = image

    # Indices for output image
    x_out = y_out = -1
    # Iterate through image
    for y in range(yKernUp, imagePadded.shape[0], strides):
        # START TODO ###################
<<<<<<< HEAD
        if y + yKernShape >= image.shape[1]:
            break
=======
        # Exit Convolution before y is out of bounds
        # END TODO ###################
        
        # START TODO ###################
<<<<<<< HEAD
        for x in range(0, image.shape[0] - xKernShape, strides):
            oXstart = x // strides
            oYstart = y // strides
            output[oXstart, oYstart] = np.sum(imagePadded[x:x+xKernShape, y:y+yKernShape] * kernel)
        y += strides

=======
        # iterate over columns and perform convolution
        # position the center of the kernel at x,y
        # and save the sum of the elementwise multiplication
        # to the corresponding pixel in the output image
        # END TODO ###################
    return output


if __name__ == '__main__':
    # Grayscale Image
    image = cv2.imread('image.png', 0)

    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=2, strides=2)
    cv2.imwrite('2DConvolved.png', output)
