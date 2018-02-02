source_shape = source_patches.shape
if best_D is None:
    cur_D = np.zeros(source_patches.shape[0:2])
    cur_D.fill(255)
else:
    cur_D = best_D

if not random_enabled:
    u_len = np.ceil(np.divide(np.log(np.divide(1, float(w))), np.log(alpha))).astype(np.int64)

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
        source[np.isnan(source)] = np.nan

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

            if 0 <= i + direction and i + direction < source_shape[0] and 0 <= j + direction and j + direction < \
                    source_shape[1]:
                if not (new_f[i + direction, j][0] + i == target_patches.shape[0]) and not (
                                new_f[i + direction, j][1] + j == target_patches.shape[1]):
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
                    dist_1 = current_distance
                else:
                    dist_1 = 1000000

                if not (new_f[i, j + direction][0] + i == target_patches.shape[0]) and not (
                                new_f[i, j + direction][1] + j == target_patches.shape[1]):

                    v_temp = new_f[i, j + direction]
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

                    dist_2 = current_distance
                else:
                    dist_2 = 1000000

                if cur_D[i, j] >= dist_1 or cur_D[i, j] >= dist_2:
                    # if cur_D[i, j] > cur_D[i + direction, j] or cur_D[i, j] > cur_D[i, j + direction]:
                    if dist_1 < dist_2:
                        cur_D[i, j] = dist_1
                        new_f[i, j] = new_f[i + direction, j]
                    else:
                        cur_D[i, j] = dist_2
                        new_f[i, j] = new_f[i, j + direction]
                elif dist_1 == 0:
                    cur_D[i, j] = dist_1
                    new_f[i, j] = new_f[i + direction, j]
                elif dist_2 == 0:
                    cur_D[i, j] = dist_2
                    new_f[i, j] = new_f[i, j + direction]

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

            if min(dist_val) < cur_D[i, j]:
                index = np.argmin(dist_val)
                x = u_x[index]
                y = u_y[index]
                cur_D[i, j] = min(dist_val)

                new_f[i, j] = y - i, x - j
            else:
                new_f[i, j] = cur_y - i, cur_x - j
best_D = cur_D
print(np.average(best_D))
##################################################################################################
##################################################################################################

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
            source[np.isnan(source)] = 0

            if best_D is None:
                cur_target = target_patches[cur_y, cur_x, ...]
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

                if 0 <= i + direction and i + direction < source_shape[0] and 0 <= j + direction and j + direction < \
                        source_shape[1]:
                    if cur_D[i, j] > cur_D[i + direction, j] or cur_D[i, j] > cur_D[i, j + direction]:
                        if cur_D[i + direction, j] < cur_D[i, j + direction]:

                            if not (new_f[i + direction, j][0] + i == target_patches.shape[0]) and not (new_f[i + direction, j][1] + j == target_patches.shape[1]):
                                v_temp = new_f[i + direction, j]
                                temp_y = v_temp[0] + i
                                temp_x = v_temp[1] + j
                                cur_target = target_patches[temp_y, temp_x, ...]
                                cur_target[np.isnan(cur_target)] = 0
                                cur_dist = np.subtract(cur_target, source)
                                cur_dist = np.abs(cur_dist)
                                if np.isnan(cur_dist).any():
                                    current_distance = np.nanmean(cur_dist)
                                else:
                                    current_distance = cur_dist.mean()

                                if current_distance < cur_D[i,j]:
                                    cur_D[i, j] = current_distance
                                    new_f[i, j] = new_f[i + direction, j]
                        else:
                            if not (new_f[i, j + direction][0] + i == target_patches.shape[0]) and  not (new_f[i, j + direction][1] + j == target_patches.shape[1]):
                                v_temp = new_f[i, j + direction]
                                temp_y = v_temp[0] + i
                                temp_x = v_temp[1] + j
                                cur_target = target_patches[temp_y, temp_x, ...]
                                cur_target[np.isnan(cur_target)] = 0
                                cur_dist = np.subtract(cur_target, source)
                                cur_dist = np.abs(cur_dist)
                                if np.isnan(cur_dist).any():
                                    current_distance = np.nanmean(cur_dist)
                                else:
                                    current_distance = cur_dist.mean()

                                if current_distance < cur_D[i, j]:
                                    cur_D[i, j] = current_distance
                                    new_f[i, j] = new_f[i, j + direction]
            v_0 = new_f[i, j]
            cur_y = v_0[0] + i
            cur_x = v_0[1] + j

            source = source_patches[i, j, ::]

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
                target[np.isnan(target)] = 0

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
    print(np.average(best_D))