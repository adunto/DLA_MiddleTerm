import numpy as np

# 데이터셋에 대하여 평균, 표준편차 산출하여 적용
def calculate_normalize(dataset):
    # 데이터셋의 axis=1, 2 에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])

    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # 데이터셋의 axis=1, 2 에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1,2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
