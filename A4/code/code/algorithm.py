# CSC320 Winter 2017
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np

# basic numpy configuration

# set random seed
import sys

np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    source_shape = source_patches.shape
    if best_D is None:
        cur_D = np.zeros(source_patches.shape[0:2])
        cur_D.fill(255)
    else:
        cur_D = best_D

    if not random_enabled:
        u_len = np.ceil(np.divide(np.log(np.divide(1,float(w))), np.log(alpha))).astype(np.int64)

        nnf_y = np.zeros(u_len)
        nnf_x = np.zeros(u_len)

        alpha_vec = np.zeros(u_len)
        alpha_vec.fill(alpha)

        w_vec = np.zeros(u_len)
        w_vec.fill(w)

        range_const = np.arange(0, u_len, 1)

        const = np.power(alpha_vec, range_const)

    if odd_iteration:
        range_y = range(source_shape[0])[::-1]
        range_x = range(source_shape[1])[::-1]
    else:
        range_y = range(source_shape[0])
        range_x = range(source_shape[1])

    for i in range_y:
        for j in range_x:

            v_0 = new_f[i, j]
            cur_y = v_0[0] + i
            cur_x = v_0[1] + j

            source = source_patches[i, j, ::]

            if best_D is None:
                cur_target = target_patches[cur_y, cur_x, ...]
                cur_target[np.isnan(cur_target)] = np.nan
                cur_dist = np.abs(np.subtract(cur_target, source))
                if np.isnan(cur_dist).any():
                    current_distance = np.nanmean(cur_dist)
                else:
                    current_distance = cur_dist.mean()
                cur_D[i, j] = current_distance

            if not propagation_enabled:
                if odd_iteration:
                    direction = 1
                else:
                    direction = -1

                if 0 <= i + direction < source_shape[0] and 0 <= j + direction < source_shape[1]:
                    if (new_f[i + direction, j][0] + i >= target_patches.shape[0]) or (
                            new_f[i + direction, j][1] + j >= target_patches.shape[1]):
                        v_temp = new_f[i + direction, j]
                        temp_y = v_temp[0] + i
                        temp_x = v_temp[1] + j
                        if new_f[i + direction, j][0] + i >= target_patches.shape[0]:
                            temp_y = target_patches.shape[0] - 1
                        if new_f[i + direction, j][1] + j >= target_patches.shape[1]:
                            temp_x = target_patches.shape[1] - 1

                        cur_target = target_patches[temp_y, temp_x, ...]
                        cur_target[np.isnan(cur_target)] = np.nan
                        cur_dist = np.subtract(cur_target, source)
                        cur_dist = np.abs(cur_dist)
                        if np.isnan(cur_dist).any():
                            current_distance = np.nanmean(cur_dist)
                        else:
                            current_distance = cur_dist.mean()
                    else:
                        v_temp = new_f[i + direction, j]
                        temp_y = v_temp[0] + i
                        temp_x = v_temp[1] + j
                        cur_target = target_patches[temp_y, temp_x, ...]
                        cur_target[np.isnan(cur_target)] = np.nan
                        cur_dist = np.subtract(cur_target, source)
                        cur_dist = np.abs(cur_dist)
                        if np.isnan(cur_dist).any():
                            current_distance = np.nanmean(cur_dist)
                        else:
                            current_distance = cur_dist.mean()

                    dist_1_x = temp_x
                    dist_1_y = temp_y

                    dist_1 = current_distance

                    if new_f[i, j + direction][0] + i >= target_patches.shape[0] or (
                            new_f[i, j + direction][1] + j >= target_patches.shape[1]):
                        v_temp = new_f[i, j + direction]
                        temp_y = v_temp[0] + i
                        temp_x = v_temp[1] + j
                        if new_f[i, j + direction][0] + i >= target_patches.shape[0]:
                            temp_y = target_patches.shape[0] - 1

                        if new_f[i, j + direction][1] + j >= target_patches.shape[1]:
                            temp_x = target_patches.shape[1] - 1

                        cur_target = target_patches[temp_y, temp_x, ...]
                        cur_dist = np.abs(np.subtract(cur_target, source))
                        if np.isnan(cur_dist).any():
                            current_distance = np.nanmean(cur_dist)
                        else:
                            current_distance = cur_dist.mean()
                    else:
                        v_temp = new_f[i, j + direction]
                        temp_y = v_temp[0] + i
                        temp_x = v_temp[1] + j
                        cur_target = target_patches[temp_y, temp_x, ...]
                        cur_dist = np.abs(np.subtract(cur_target, source))
                        if np.isnan(cur_dist).any():
                            current_distance = np.nanmean(cur_dist)
                        else:
                            current_distance = cur_dist.mean()
                    dist_2_x = temp_x
                    dist_2_y = temp_y

                    dist_2 = current_distance

                    if cur_D[i, j] > dist_1 or cur_D[i,j] > dist_2:
                        if dist_1 < dist_2:
                            cur_D[i, j] = dist_1
                            new_f[i, j] = dist_1_y - i, dist_1_x - j
                        else:
                            cur_D[i, j] = dist_2
                            new_f[i, j] = dist_2_y - i, dist_2_x - j

            v_0 = new_f[i, j]
            cur_y = v_0[0] + i
            cur_x = v_0[1] + j

            source = source_patches[i, j, ::]
            source[np.isnan(source)] = np.nan

            if not random_enabled:
                if v_0[0] < 0:
                    v_0[0] += target_patches.shape[0]
                if v_0[1] < 0:
                    v_0[1] += target_patches.shape[1]

                nnf_y.fill(v_0[0])
                nnf_x.fill(v_0[1])

                random_y = np.random.randint(low=-v_0[0], high=target_patches.shape[0] - v_0[0], size=u_len)
                random_x = np.random.randint(low=-v_0[1], high=target_patches.shape[1] - v_0[1], size=u_len)

                u_y = np.round(np.add(nnf_y, np.multiply(const, random_y)))
                u_x = np.round(np.add(nnf_x, np.multiply(const, random_x)))

                target = target_patches[u_y.astype(np.int64), u_x.astype(np.int64), ...]
                target[np.isnan(target)] = np.nan

                tile_source = np.tile(source, (10, 1, 1))

                dist = np.subtract(target, tile_source)
                dist = np.abs(dist)

                dist_val = np.zeros(u_len)

                for k in range(u_len):
                    if np.isnan(dist[k]).any():
                        dist_val[k] = np.nanmean(dist[k])
                    else:
                        dist_val[k] = dist[k].mean()

                if min(dist_val) < cur_D[i,j]:
                    index = np.argmin(dist_val)
                    x = u_x[index]
                    y = u_y[index]
                    cur_D[i, j] = min(dist_val)

                    new_f[i,j] = y - i, x - j
                else:
                    new_f[i,j] = cur_y - i, cur_x - j
    best_D = cur_D

    #############################################

    return new_f, best_D, global_vars


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    tar_shape = target.shape
    coor = make_coordinates_matrix(tar_shape, step=1)

    nnf_y,nnf_x = np.dsplit(f, 2)
    coor_y, coor_x = np.dsplit(coor, 2)

    axis_y = coor_y + nnf_y
    axis_x = coor_x + nnf_x

    rec_source = target[axis_y, axis_x, ]
    rec_source = np.squeeze(rec_source)

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
