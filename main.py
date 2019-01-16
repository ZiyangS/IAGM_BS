import GMM
import glob
import cv2 as cv
from IAGMM import IAGMM
if __name__ == '__main__':
    train_num = 300
    category_path = 'baseline'
    video_path = 'pedestrians'
    dataset_path = r'./dataset/' + category_path + '/' + video_path + '/input'
    results_path = r'./results/' + category_path + '/' + video_path

    gmm = GMM.GMM(data_dir=dataset_path, train_num=train_num)
    gmm.train()
    print('train finished')
    file_list = glob.glob(dataset_path + '/*')
    for index, file in enumerate(file_list):
        # only the frames after train_num will be used to evaulate scores.
        if index + 1 < train_num:
            continue
        print('infering:{}'.format(file))
        img = cv.imread(file)
        img = gmm.infer(img)
        cv.imwrite(results_path +'/in%06d'%(index+1)+'.jpg', img)
