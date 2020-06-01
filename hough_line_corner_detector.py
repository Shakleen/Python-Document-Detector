from math import sin, cos
import cv2
import numpy as np
from matplotlib import pyplot as plt
from processors import Opener, Closer, EdgeDetector
from sklearn.cluster import KMeans
from itertools import combinations


class HoughLineParams:
    def __init__(self, rho_acc = 1, theta_acc = 180, thresh = 100):
        self.rho_acc = rho_acc
        self.theta_acc = theta_acc
        self.thresh = thresh


class HoughLineCornerDetector:
    def __init__(self, hough_line_params, output_process = True):
        assert isinstance(hough_line_params, HoughLineParams), "Invalid type for hough_line_params"
        self._hough_line_params = hough_line_params
        self.output_process = output_process
        self._preprocessor = [
            Closer(output_process = output_process), 
            Opener(output_process = output_process), 
            EdgeDetector(output_process = output_process)
        ]

    
    def __call__(self, image):
        # Step 1: Process for edge detection
        self._image = image
        for processor in self._preprocessor:
            self._image = processor(self._image)
        
        # Step 2: Get hough lines
        self._lines = self._get_edge_lines()

        # Step 3: Get intersection points
        return self._get_intersections()

    
    def _get_edge_lines(self):
        # Step 1: Get all the hough lines
        lines = cv2.HoughLines(
            self._image, 
            self._hough_line_params.rho_acc, 
            np.pi / self._hough_line_params.theta_acc, 
            self._hough_line_params.thresh
        )
        if self.output_process: self._draw_hough_lines(lines)

        # Step 2: Cluster hough lines into 4 groups using kmeans
        kmeans = self._cluster_lines(lines)
        if self.output_process: self._draw_cluster_lines(lines, kmeans)

        # Step 3: Returns cluster centers
        return [[center.tolist()] for center in kmeans.cluster_centers_]

    
    def _draw_hough_lines(self, lines):
        hough_line_output = self._get_color_image()

        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            n = 5000
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(
                hough_line_output, 
                (x1, y1), 
                (x2, y2), 
                (0, 0, 255), 
                2
            )

        cv2.imwrite('output/hough_line.jpg', hough_line_output)


    def _cluster_lines(self, lines):
        X = np.array([[line[0][0], line[0][1]] for line in lines])
        kmeans = KMeans(
            n_clusters = 4, 
            init = 'k-means++', 
            max_iter = 300, 
            n_init = 10, 
            random_state = 0
        ).fit(X)
        
        return kmeans


    def _draw_cluster_lines(self, lines, kmeans):
        grouped_output = self._get_color_image()
        LABEL_COLOR_MAP = {
            0 : (255, 0, 0),
            1 : (0, 255, 0),
            2 : (0, 0, 255),
            3 : (255, 0, 255)
        }
        label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]

        for idx, line in enumerate(lines):
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            n = 5000
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(
                grouped_output, 
                (x1, y1), 
                (x2, y2), 
                label_color[idx], 
                2
            )

        cv2.imwrite('output/grouped.jpg', grouped_output)

    
    def _get_intersections(self):
        """Finds the intersections between groups of lines."""
        lines = self._lines
        group_pair = combinations(range(0, 4), 2)

        intersections = []

        x_in_range = lambda x: 0 <= x <= self._image.shape[1]
        y_in_range = lambda y: 0 <= y <= self._image.shape[0]

        for i, j in group_pair:
            for line1 in lines[i]:
                for line2 in lines[j]:
                    int_point = self._intersection(line1, line2)
                    
                    if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]): 
                        intersections.append(int_point)


        if self.output_process: self._draw_intersections(intersections)

        return intersections

    
    def _intersection(self, line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])

        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]


    def _draw_intersections(self, intersections):
        intersection_point_output = self._image.copy()

        for line in self._lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            n = 5000
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(
                intersection_point_output, 
                (x1, y1), 
                (x2, y2), 
                (0, 0, 255), 
                2
            )

        for point in intersections:
            x, y = point[0]

            cv2.circle(
                intersection_point_output,
                (x, y),
                15,
                (255, 255, 127),
                15
            )

        cv2.imwrite('output/intersection_point_output.jpg', intersection_point_output)


    def _get_color_image(self):
        return cv2.cvtColor(self._image.copy(), cv2.COLOR_GRAY2RGB)