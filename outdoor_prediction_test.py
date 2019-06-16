import unittest2
import cv2
import numpy as np

import outdoor_prediction as op


class TestIndoorOutdoor(unittest2.TestCase):

    def test_image_resizing(self):
        img = cv2.imread("./test_data/big_img.jpg")
        resized = op.make_square(img)
        exp_resized = cv2.imread("./test_data/small_img.jpg")
        mse = np.mean((exp_resized - resized) ** 2)
        self.assertLess(mse, 4)

    def test_prediction_indoor(self):
        model = op.load_prediction_model("./models/weights.h5")
        img = cv2.imread("./test_data/00000.jpg")
        img = op.make_square(img)
        prediction = op.make_prediction(model, np.array([img]))
        self.assertEqual(prediction[0][0], "Indoor")

    def test_prediction_outdoor(self):
        model = op.load_prediction_model("./models/weights.h5")
        img = cv2.imread("./test_data/00026.jpg")
        img = op.make_square(img)
        prediction = op.make_prediction(model, np.array([img]))
        self.assertEqual(prediction[0][0], "Outdoor")

    def test_convert_folder(self):
        data = op.convert_folder_into_data("./test_data/test_folder/")
        self.assertEqual(data.shape, (2, 224, 224, 3))

    def test_predict_file(self):
        model = op.load_prediction_model("./models/weights.h5")
        labels = op.predict_for_files_with_model(model, "./test_data/big_img.jpg")
        self.assertEqual(labels, [['Outdoor']])

    def test_predict_folder(self):
        model = op.load_prediction_model("./models/weights.h5")
        labels = op.predict_for_files_with_model(model, "./test_data/test_folder/")
        self.assertItemsEqual(labels, [['Indoor'], ['Outdoor']])


if __name__ == '__main__':
    unittest2.main()
