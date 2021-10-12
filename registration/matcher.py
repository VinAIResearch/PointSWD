import numpy as np
import open3d


class Matcher:
    def __init__(self, threshold=0.075, fitness_thresh=0.3, rmse_thresh=0.2):
        super().__init__()
        self.threshold = threshold
        self.fitness_thresh = fitness_thresh
        self.rmse_thresh = rmse_thresh

    def compute_transformation(self, source_points, target_points, source_descriptors, target_descriptors):
        source_features = open3d.registration.Feature()
        source_features.data = np.transpose(source_descriptors)
        target_features = open3d.registration.Feature()
        target_features.data = np.transpose(target_descriptors)

        processed_source_points = open3d.geometry.PointCloud()
        processed_source_points.points = open3d.utility.Vector3dVector(source_points)
        processed_target_points = open3d.geometry.PointCloud()
        processed_target_points.points = open3d.utility.Vector3dVector(target_points)

        result = open3d.registration.registration_ransac_based_on_feature_matching(
            processed_source_points,
            processed_target_points,
            source_features,
            target_features,
            self.threshold,
            open3d.registration.TransformationEstimationPointToPoint(False),
            4,
            [open3d.registration.CorrespondenceCheckerBasedOnDistance(self.threshold)],
            open3d.registration.RANSACConvergenceCriteria(4000000, 500),
        )

        information = open3d.registration.get_information_matrix_from_point_clouds(
            processed_source_points, processed_target_points, self.threshold, result.transformation
        )
        threshold_num_point = min(len(processed_source_points.points), len(processed_target_points.points))

        fitness = information[5, 5] / float(threshold_num_point)

        if (fitness > self.fitness_thresh) and (result.inlier_rmse < self.rmse_thresh):
            check = True
        else:
            check = False
        return result.transformation, check
