from IAGM import IAGM
import cv2 as cv

if __name__ == '__main__':
    # train_num = 470
    # test_num = 1700
    # category_path = 'baseline'
    # video_path = 'highway'

    # train_num = 570
    # test_num = 2050
    # category_path = 'baseline'
    # video_path = 'office'

    # train_num = 300
    # test_num = 1099
    # category_path = 'baseline'
    # video_path = 'pedestrians'

    #  train_num = 300
    # test_num = 1200
    # category_path = 'baseline'
    # video_path = 'PETS2006'

    # train_num = 400
    # test_num = 1184
    # category_path = 'dynamicBackground'
    # video_path = 'fountain01'

    # train_num = 1000
    # test_num = 4000
    # category_path = 'dynamicBackground'
    # video_path = 'fall'

    # train_num = 1900
    # test_num = 7999
    # category_path = 'dynamicBackground'
    # video_path = 'boats'

    # train_num = 790
    # test_num = 2500
    # category_path = 'cameraJitter'
    # video_path = 'boulevard'

    # train_num = 900
    # test_num = 1570
    # category_path = 'cameraJitter'
    # video_path = 'traffic'

    train_num = 175
    test_num = 3200
    category_path = 'intermittentObjectMotion'
    video_path = 'streetLight'

    # train_num = 1320
    # test_num = 3200
    # category_path = 'intermittentObjectMotion'
    # video_path = 'tramstop'

    # train_num = 1100
    # test_num = 2500
    # category_path = 'intermittentObjectMotion'
    # video_path = 'parking'

    # train_num = 2450
    # test_num = 4500
    # category_path = 'intermittentObjectMotion'
    # video_path = 'abandonedBox'

    # train_num = 500
    # test_num = 2750
    # category_path = 'intermittentObjectMotion'
    # video_path = 'sofa'

    # train_num = 1000
    # test_num = 2500
    # category_path = 'intermittentObjectMotion'
    # video_path = 'winterDriveway'

    # train_num = 800
    # test_num = 1200
    # category_path = 'cameraJitter'
    # video_path = 'sidewalk'

    # train_num = 400
    # test_num = 2000
    # category_path = 'shadow'
    # video_path = 'backdoor'

    # train_num = 500
    # test_num = 3400
    # category_path = 'shadow'
    # video_path = 'copymachine'

    # train_num = 600
    # test_num = 4900
    # category_path = 'thermal'
    # video_path = 'library'
    # #
    # train_num = 1000
    # test_num = 6500
    # category_path = 'thermal'
    # video_path = 'lakeside'
    #
    # train_num = 1100
    # test_num = 7400
    # category_path = 'shadow'
    # video_path = 'cubicle'

    # train_num = 250
    # test_num = 600
    # category_path = 'thermal'
    # video_path = 'park'

    # train_num = 1000
    # test_num = 6500
    # category_path = 'thermal'
    # video_path = 'lakeSide'

    dataset_path = r'./dataset/' + category_path + '/' + video_path + '/input'
    results_path = r'./results/' + category_path + '/' + video_path
    ROI_path = r'./dataset/' + category_path + '/' + video_path + '/ROI.bmp'

    iagm = IAGM(data_dir=dataset_path, results_path=results_path, ROI_path=ROI_path, train_num=train_num, test_num=test_num)
    iagm.model_training()