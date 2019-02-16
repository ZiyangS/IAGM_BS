import os
import glob
import time
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    train_number = 300
    evaluate_end = 1099
    evaluate_start = train_number

    category_path = 'baseline'
    video_path = 'pedestrians'

    gt_path = r'./dataset/' + category_path + '/' + video_path + '/groundtruth'
    results_path = r'./results/' + category_path + '/' + video_path

    recall_list = []
    precision_list = []

    # print(gt_img.shape) 240*360
    highest_fg_number = 0
    cum_fg_number = 0
    for index in range(evaluate_start, evaluate_end+1):
        gt_file_name = os.path.join(gt_path, 'gt%06d' % index + '.png')
        gt_img = cv.imread(gt_file_name)
        results_file_name = os.path.join(results_path +'/in%06d' % index + '.jpg')
        results_img = cv.imread(results_file_name)
        gt_fg_num = 0
        detected_fg_num = 0
        correctly_identified_fg_num = 0
        for i in range(gt_img.shape[0]):
            for j in range(gt_img.shape[0]):
                if np.sum(gt_img[i][j]) != 0:
                    gt_fg_num += 1
                if np.sum(results_img[i][j]) != 765:
                    detected_fg_num += 1
                if (np.sum(gt_img[i][j]) != 0) and (np.sum(results_img[i][j] != 765)):
                    correctly_identified_fg_num += 1
        if highest_fg_number < gt_fg_num:
            highest_fg_number = gt_fg_num
        if gt_fg_num != 0:
            cum_fg_number += gt_fg_num
            print(correctly_identified_fg_num)
            print(gt_fg_num)
            print(detected_fg_num)
            recall = correctly_identified_fg_num / gt_fg_num
            print(recall)
            precision = correctly_identified_fg_num / detected_fg_num
            print(precision)
            recall_list.append(recall)
            precision_list.append(precision)
            print("____")
    print(np.average(recall))
    print(np.average(precision))
    print(highest_fg_number)
    print(cum_fg_number)
