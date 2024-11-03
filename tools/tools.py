import torch.nn.functional as ff
import torch

def matrix_to_list(matrix):
    return [item for row in matrix for item in row]

def flip_matrix_horizontally(data, n, m):
    matrix = [data[i * m:(i + 1) * m] for i in range(n)]
    flipped_matrix = [row[::-1] for row in matrix]
    return matrix_to_list(flipped_matrix)

def flip_matrix_vertically(data, n, m):
    matrix = [data[i * m:(i + 1) * m] for i in range(n)]
    flipped_matrix = matrix[::-1]
    return matrix_to_list(flipped_matrix)

def spiral_traverse(n, m):
    result = []
    if not n or not m:
        return result

    left, right, top, bottom = 0, m - 1, 0, n - 1

    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            result.append((top, i))
        top += 1

        for i in range(top, bottom + 1):
            result.append((i, right))
        right -= 1

        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append((bottom, i))
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append((i, left))
            left += 1
    return result

def spiral_traverse_ccw(n, m):
    result = []
    if not n or not m:
        return result

    left, right, top, bottom = 0, m - 1, 0, n - 1

    while left <= right and top <= bottom:
        for i in range(top, bottom + 1):
            result.append((i, left))
        left += 1
        
        if top <= bottom: 
            for i in range(left, right + 1):
                result.append((bottom, i))
            bottom -= 1
        
        if left <= right: 
            for i in range(bottom, top - 1, -1):
                result.append((i, right))
            right -= 1

        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append((top, i))
            top += 1
    return result


def spiral(n, m):
    result_list = []

    spiral_order = spiral_traverse(n, m)
    order_list = [0] * (n * m)
    for i, (x, y) in enumerate(spiral_order):
        order_list[x * m + y] = i

    result_list.append(order_list)
    re_order_list = [n * m - i - 1 for i in order_list]
    result_list.append(re_order_list)

    result_list.append(flip_matrix_horizontally(order_list, n, m))
    result_list.append(flip_matrix_horizontally(re_order_list, n, m))

    result_list.append(flip_matrix_vertically(order_list, n, m))
    result_list.append(flip_matrix_vertically(re_order_list, n, m))

    result_list.append(flip_matrix_vertically(flip_matrix_horizontally(order_list, n, m), n, m))
    result_list.append(flip_matrix_vertically(flip_matrix_horizontally(re_order_list, n, m), n, m))

    spiral_order_ccw = spiral_traverse_ccw(n, m)
    order_list = [0] * (n * m)
    for i, (x, y) in enumerate(spiral_order_ccw):
        order_list[x * m + y] = i

    result_list.append(order_list)
    re_order_list = [n * m - i - 1 for i in order_list]
    result_list.append(re_order_list)

    result_list.append(flip_matrix_horizontally(order_list, n, m))
    result_list.append(flip_matrix_horizontally(re_order_list, n, m))

    result_list.append(flip_matrix_vertically(order_list, n, m))
    result_list.append(flip_matrix_vertically(re_order_list, n, m))

    result_list.append(flip_matrix_vertically(flip_matrix_horizontally(order_list, n, m), n, m))
    result_list.append(flip_matrix_vertically(flip_matrix_horizontally(re_order_list, n, m), n, m))

    original_order_indexes_list = []
    for k in range(len(result_list)):
        index_mapping = {index: i for i, index in enumerate(result_list[k])}
        original_order_indexes = [index_mapping[i] for i in range(len(result_list[k]))]
        original_order_indexes_list.append(original_order_indexes)

    return result_list, original_order_indexes_list





def calculate_conv1d_parameters(D, K):
    for kernel_size in range(1, D + 1):
        for stride in range(1, D + 1):
            for padding in range(0, D + 1):
                output_size = (D + 2 * padding - kernel_size) // stride + 1
                if output_size == K:
                    return kernel_size, stride, padding
    return None, None, None



def avg_pool_features(x, K):
    B, F, D = x.shape
    assert D % K == 0, 'D must be divisible by K'
    stride = D // K
    kernel_size = D - (K - 1) * stride
    return ff.avg_pool1d(x, kernel_size=kernel_size, stride=stride)


def covariance_loss(A, B):
    B, F, D = A.shape 
    A_mean = A.mean(dim=2, keepdim=True)
    B_mean = B.mean(dim=2, keepdim=True)
    A_centered = A - A_mean
    B_centered = B - B_mean
    covariances = []
    for i in range(B):
        cov_per_frame = torch.bmm(A_centered[i].transpose(0, 1), B_centered[i]) / (D - 1)
        covariances.append(cov_per_frame)
    covariances = torch.stack(covariances) 
    loss = -torch.mean(covariances)
    return loss
