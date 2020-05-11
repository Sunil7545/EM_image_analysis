from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm
from os import listdir
from os.path import isfile, join


# OpenCv window to display the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 960, 645)


def threshold_image(gray_image, name_bw, threshold):
    """
    This computes the binary image of the input image using a threshold

    :param gray_image: input image
    :param threshold: input threshold
    :param name_bw: name of the binary image
    :return: BW image
    """

    # perform Gaussian blurring to remove unwanted noisy components
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # convert the smooth image into a bw image
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]

    # perform morphological operation to remove small components
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=1)

    # store the bw image
    cv2.imwrite("threshold_" + name_bw, thresh)
    return thresh


def compute_contour(thresh):
    """
    This method computes the contour of the particles available in the EM images

    :param thresh: input binary image
    :return: contours of the particles
    """
    # Label connected regions of the input image.
    # Two pixels are connected when they are neighbors and have the same value.
    # background pixels are labeled as 0
    labels = measure.label(thresh, neighbors=8, background=0)

    # create a mask image of the size of input image
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique labels of particles
    for label in np.unique(labels):
        # check for the background pixel
        if label == 0:
            continue

        # otherwise, construct the label mask and count the number of pixels
        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255

        # check the number of pixels in that particular label
        num_pixels = cv2.countNonZero(label_mask)

        # if the number of pixels in that particular particle  is sufficiently large then keep it
        # and that is defined by user"
        if num_pixels > 0:
            mask = cv2.add(mask, label_mask)

    # Finds contours of particles from the binary mask image. The mode is set to
    # retrieves only the extreme outer contours an uses simple chain approximation to store the contour points
    contours_image = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # grab the particle contours
    contours_image = imutils.grab_contours(contours_image)

    # sort the particle contours
    contours_image = contours.sort_contours(contours_image)[0]

    return contours_image


def hist_plot(data_array, label, hist_title, f_name):
    fig = plt.figure()
    num_bin = 20
    # the histogram of the data
    n, bins, patches = plt.hist(data_array, num_bin, facecolor='green', alpha=0.5)
    # best fit of data
    (mu, sigma) = norm.fit(data_array)
    # add a 'best fit' line
    # y = norm.pdf(bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=2)
    # plot
    plt.title(hist_title)
    plt.xlabel(label)
    plt.ylabel('frequency')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.grid(True)
    plt.savefig('hist_' + hist_title + f_name)


def compute_agglomerate_distance(agglomerate_points):
    """
    This method computes distances between the agglomerates particles

    :param agglomerate_points: input points
    :return: distances
    """
    # compute the shape of the input points
    shape_points = agglomerate_points.shape

    # array to store distances
    agglomerate_distance = np.zeros([shape_points[0], 1], dtype=float)

    # go through each agglomerate point
    for i in range(shape_points[0]):

        # get the focus point
        focus_point = agglomerate_points[i, :]

        # difference and between other points
        difference = agglomerate_points - focus_point
        distance_vec = np.linalg.norm(difference, axis=1)
        agglomerate_distance[i, 0] = np.sum(distance_vec)/(shape_points[0]-1)

    return agglomerate_distance


def compute_particle_params(contours_image_input, image_input, hist_name):
    """
    This method computes several shape related parameters from the contours of the particles.

    :param contours_image_input: input contour image
    :param image_input: original image
    :param hist_name: the histogram name
    :return:
    """

    # check the number of particles from contour image
    num_particle = len(contours_image_input)

    # this parameter computes the maximum length of a particle i.e. maximum distance between two point of a contour
    max_length_particle = np.zeros([num_particle, 1], dtype=float)

    # this parameter computes the areas of  particles
    area_particle = np.zeros([num_particle, 1], dtype=float)

    # this parameter computes the center of the mass of  particles
    centroid_particle = np.zeros([num_particle, 2], dtype=float)

    # this parameter computes the ratio between maximum and minimum length of a particle
    aspect_ratio_particle = np.zeros([num_particle, 1], dtype=float)

    # this parameter computes the perimeter of particles
    perimeter_particle = np.zeros([num_particle, 1], dtype=float)

    # this parameter computes the perimeter of particles
    circularity_particle = np.zeros([num_particle, 1], dtype=float)

    # these are the bigger particles in the EM image
    agglomerate__points = []

    # corresponding areas
    agglomerate__area = []

    # loop over all the contours
    for (i, c) in enumerate(contours_image_input):

        # compute the moment of a particular contours
        moment_contour = cv2.moments(c)

        # compute the centroid of the particle using the moment
        centroid_particle[i, 0] = moment_contour['m10'] / moment_contour['m00']
        centroid_particle[i, 1] = moment_contour['m01'] / moment_contour['m00']

        # compute the area of the particle
        area_particle[i, 0] = cv2.contourArea(c)

        # compute the perimeter of the particle
        perimeter_particle[i, 0] = cv2.arcLength(c, True)

        # compute the minimum size of rectangle to fit the corresponding particle particle.
        # The width and height will be maximum and minimum distance between the two pints of the corresponding particle
        min_rectangle = cv2.minAreaRect(c)
        width_height = min_rectangle[1]

        # compute the aspect ratio using the rectangle info
        aspect_ratio_particle[i, 0] = width_height[1] / width_height[0]
        box = cv2.boxPoints(min_rectangle)
        box = np.int0(box)

        # a circle which encloses the particle
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        # select the agglomerate, the particles which are bigger than 100 pixels
        if area_particle[i, 0] > 100:
            # a circle to highlight the agglomerate
            image_input = cv2.circle(image_input, (int(cX), int(cY)), int(radius), (0, 255, 0), 2)
            agglomerate__points.append(centroid_particle[i, :])

            # area of agglomerate
            agglomerate__area.append(area_particle[i, 0])

        # maximum length of each particle
        max_length_particle[i, 0] = 2 * radius

        # circularity of the particle, if it is 1 then perfect circle
        circularity_particle[i, 0] = area_particle[i, 0] / (np.pi*radius*radius)

        # Highlight the agglomerate
        cv2.imshow("image", image_input)

    # For the visualization purpose, I select one agglomerate
    # and compute the distance between this and neighbour agglomerate
    focus_point = agglomerate__points[5]

    # select 7 agglomerate for visual purpose only
    for i in range(7):

        # skip if it is the center agglomerate
        if i == 5:
            continue

        # compute the distance between neighbour and center agglomerate using numpy norm
        difference = agglomerate__points[i] - focus_point
        distance_vec = np.round(np.linalg.norm(difference), 2)

        # draw a line between these two agglomerates
        cv2.line(image_input, (int(focus_point[0]), int(focus_point[1])),
                 (int(agglomerate__points[i][0]), int(agglomerate__points[i][1])),
                 (0, 100, 200), 2)

        # write the distance in the image
        cv2.putText(image_input, str(distance_vec), tuple((focus_point/2).astype(np.int) +
                                                          (agglomerate__points[i]/2).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.85, (255, 0, 0), 1, cv2.LINE_AA)

    # compute distances between all agglomerate
    distance_agglomerate = compute_agglomerate_distance(np.array(agglomerate__points))
    cv2.imwrite('particle_' + hist_name, image_input)
    hist_plot(distance_agglomerate, 'Distance', 'Agglomerate Distance (pixels)', hist_name)


only_files = [f for f in listdir('input_files') if isfile(join('input_files', f))]
for jj in range(len(only_files)):
    file_name = 'input_files\\' + only_files[jj]
    image = cv2.imread(file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    first_input = threshold_image(gray, only_files[jj], threshold=170)
    particles = compute_contour(first_input)
    compute_particle_params(particles, image, 'agglo.jpg')

# plt.show()
# cv2.waitKey()
