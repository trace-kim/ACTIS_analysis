/* One block owns one aperture. All pixel coordinates below are tile-local. */
__device__ void sort_block(float *a, int size) {
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j; j >>= 1) {
            for (int i = threadIdx.x; i < size; i += blockDim.x) {
                int other = i ^ j;
                if (other > i && ((i & k) ? a[i] < a[other] : a[i] > a[other])) {
                    float v = a[i];
                    a[i] = a[other];
                    a[other] = v;
                }
            }
            __syncthreads();
        }
    }
}

__device__ double quantile(const float *v, int n, double q) {
    double index = (n - 1) * q;
    int lo = (int)index;
    return v[lo] + (index - lo) * (v[min(lo + 1, n - 1)] - v[lo]);
}

extern "C" __global__ void contours(const float *tiles, const double *sites, const int *tile_id,
                                    int side, const double *offsets, const double *directions,
                                    int distances, int dark_start, int bright_stop, double radius,
                                    double step, int sorted_size, int shape, double *output) {
    int site = blockIdx.x, tid = threadIdx.x;
    const float *image = tiles + tile_id[site] * side * side;
    double cx = sites[2 * site], cy = sites[2 * site + 1];
    double *out = output + 3 * site;
    if (cx < radius || cy < radius || cx > side - 1 - radius || cy > side - 1 - radius) {
        if (!tid)
            out[0] = out[1] = out[2] = nan("");
        return;
    }
    const int rays = ACTIS_RAYS;
    extern __shared__ float shared[];
    float *samples = shared, *ordered = shared + rays * distances;
    __shared__ double level, amplitude, lengths[rays], vx[rays], vy[rays];
    __shared__ int resolved;
    for (int i = tid; i < rays * distances; i += blockDim.x) {
        double x = cx + offsets[2 * i], y = cy + offsets[2 * i + 1];
        // Use the same float32 interpolation stencil as CuPy map_coordinates.
        int ix = (int)floorf((float)x), iy = (int)floorf((float)y);
        int right = min(ix + 1, side - 1), bottom = min(iy + 1, side - 1);
        double fx = x - ix, fy = y - iy;
        float value = image[iy * side + ix] * (float)((1 - fy) * (1 - fx));
        value = fmaf(image[iy * side + right], (float)((1 - fy) * fx), value);
        value = fmaf(image[bottom * side + ix], (float)(fy * (1 - fx)), value);
        samples[i] = fmaf(image[bottom * side + right], (float)(fy * fx), value);
    }
    __syncthreads();
    int dark_count = rays * (distances - dark_start);
    for (int i = tid; i < sorted_size; i += blockDim.x)
        ordered[i] = i < dark_count ? samples[i / (distances - dark_start) * distances +
                                              dark_start + i % (distances - dark_start)]
                                    : __int_as_float(0x7f800000);
    __syncthreads();
    sort_block(ordered, sorted_size);
    __shared__ double dark, floor_noise;
    if (!tid) {
        dark = quantile(ordered, dark_count, .16);
        floor_noise =
            fmax(1.90 * (quantile(ordered, dark_count, .32) - dark), 1e-6 * fmax(fabs(dark), 1.0));
    }
    __syncthreads();
    int bright_count = rays * bright_stop, bright_size = 1;
    while (bright_size < bright_count)
        bright_size *= 2;
    for (int i = tid; i < bright_size; i += blockDim.x)
        ordered[i] = i < bright_count ? samples[i / bright_stop * distances + i % bright_stop]
                                      : __int_as_float(0x7f800000);
    __syncthreads();
    sort_block(ordered, bright_size);
    if (!tid) {
        amplitude = quantile(ordered, bright_count, .96) - dark;
        level = dark + .5 * amplitude;
        resolved = 0;
        out[0] = out[1] = nan("");
        out[2] = amplitude;
    }
    __syncthreads();
    if (amplitude < 3.5 * floor_noise)
        return;
    if (tid < rays) {
        const float *line = samples + tid * distances;
        double length = -1;
        if (line[0] >= level) {
            for (int j = 0; j < distances - 1; j++) {
                if (line[j] >= level && line[j + 1] < level) {
                    double frac = (line[j] - line[j + 1]) > 1e-8 * amplitude
                                      ? (line[j] - level) / (line[j] - line[j + 1])
                                      : 0;
                    length = (j + frac) * step;
                    break;
                }
            }
        }
        lengths[tid] = length;
        if (length >= 0)
            atomicAdd(&resolved, 1);
    }
    __syncthreads();
    if (resolved < .94 * rays)
        return;
    if (tid < rays) {
        double length = lengths[tid];
        if (length < 0) {
            int l = 1, r = 1;
            while (lengths[(tid - l + rays) % rays] < 0)
                ++l;
            while (lengths[(tid + r) % rays] < 0)
                ++r;
            length =
                (lengths[(tid - l + rays) % rays] * r + lengths[(tid + r) % rays] * l) / (l + r);
        }
        vx[tid] = length * directions[2 * tid];
        vy[tid] = length * directions[2 * tid + 1];
    }
    __syncthreads();
    if (!tid) {
        double area2 = 0, mx = 0, my = 0, mxx = 0, myy = 0, mxy = 0;
        for (int j = 0; j < rays; j++) {
            int k = (j + 1) % rays;
            double x = vx[j], y = vy[j], xx = vx[k], yy = vy[k], c = x * yy - xx * y;
            area2 += c;
            mx += (x + xx) * c;
            my += (y + yy) * c;
            mxx += (x * x + x * xx + xx * xx) * c;
            myy += (y * y + y * yy + yy * yy) * c;
            mxy += ((2 * x + xx) * y + (x + 2 * xx) * yy) * c;
        }
        double area = area2 / 2;
        if (area <= 0)
            return;
        mx /= 6 * area;
        my /= 6 * area;
        double a = mxx / (12 * area) - mx * mx, b = myy / (12 * area) - my * my,
               c = mxy / (24 * area) - mx * my;
        double avg = (a + b) / 2, gap = hypot((a - b) / 2, c);
        if (avg <= gap)
            return;
        out[0] = shape == 2   ? 4 * sqrt(avg + gap)
                 : shape == 1 ? sqrt(area)
                              : 2 * sqrt(area / 3.141592653589793);
        out[1] = shape == 2 ? 4 * sqrt(avg - gap) : out[0];
        out[2] = amplitude;
    }
}

extern "C" __global__ void flux(const float *tiles, const double *sites, const int *tile_id,
                                int side, double pitch, int half, int sorted_size, double *result) {
    int i = blockIdx.x, t = threadIdx.x;
    double x = sites[2 * i], y = sites[2 * i + 1], rad = pitch * .5;
    int ix = __double2int_rn(x), iy = __double2int_rn(y), width = 2 * half + 1;
    if (ix < half || iy < half || ix + half >= side || iy + half >= side) {
        if (!t)
            result[i] = nan("");
        return;
    }
    const float *image = tiles + tile_id[i] * side * side;
    extern __shared__ float ring[];
    __shared__ int n;
    __shared__ double bg, sigma;
    __shared__ double parts[5][128];
    if (!t)
        n = 0;
    __syncthreads();
    int ring_count = 0;
    for (int p = t; p < sorted_size; p += blockDim.x) {
        float value = __int_as_float(0x7f800000);
        if (p < width * width) {
            int px = ix + p % width - half, py = iy + p / width - half;
            double dx = px - x, dy = py - y, d = dx * dx + dy * dy;
            if (d >= .48 * pitch * (.48 * pitch) && d <= .64 * pitch * (.64 * pitch)) {
                value = image[py * side + px];
                ++ring_count;
            }
        }
        ring[p] = value;
    }
    atomicAdd(&n, ring_count);
    __syncthreads();
    sort_block(ring, sorted_size);
    if (!t) {
        int last = max(0, n - 1);
        bg = ring[__double2int_rn(.5 * last)];
        sigma = (ring[__double2int_rn(.75 * last)] - ring[__double2int_rn(.25 * last)]) / 1.349;
    }
    __syncthreads();
    double sum = 0, mx = 0, my = 0, ws = 0, w2 = 0;
    for (int p = t; p < width * width; p += blockDim.x) {
        int px = ix + p % width - half, py = iy + p / width - half;
        double dx = px - x, dy = py - y, d = dx * dx + dy * dy;
        double w = (d < rad * rad);
        if (d >= fmax(0., rad - .72) * fmax(0., rad - .72) && d < (rad + .72) * (rad + .72)) {
            int inside = 0;
            for (int a = 0; a < 4; a++)
                for (int b = 0; b < 4; b++) {
                    double sx = dx - .375 + .25 * a, sy = dy - .375 + .25 * b;
                    inside += (sx * sx + sy * sy <= rad * rad);
                }
            w = inside / 16.;
        }
        double v = w * (image[py * side + px] - bg);
        sum += v;
        mx += v * dx;
        my += v * dy;
        ws += w;
        w2 += w * w;
    }
    parts[0][t] = sum;
    parts[1][t] = mx;
    parts[2][t] = my;
    parts[3][t] = ws;
    parts[4][t] = w2;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (t < stride)
            for (int j = 0; j < 5; j++)
                parts[j][t] += parts[j][t + stride];
        __syncthreads();
    }
    if (!t) {
        sum = parts[0][0];
        ws = parts[3][0];
        double noise = fmax(sigma * sqrt(parts[4][0] + 1.57 * ws * ws / max(n, 1)), 1e-9);
        result[i] =
            sum > 0 && sum / noise >= 3.5 && hypot(parts[1][0], parts[2][0]) <= rad * .5 * sum
                ? sum
                : nan("");
    }
}

/* Robust aperture fit. A block owns a single native stamp; threads own pixels.
   There is one damped normal system per site and per iteration. */
__device__ float clipf(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }

struct Aperture {
    const float *data;
    int n, nh, radius;
    float w0, dw, h0, dh;
    __device__ float sample(float x, float y, float w, float h, float *deriv) const {
        float px = clipf((x + radius) * 5, 0, n - 1), py = clipf((y + radius) * 5, 0, n - 1);
        int ix = min((int)px, n - 2), iy = min((int)py, n - 2);
        float fx = px - ix, fy = py - iy;
        float wi = clipf((w - w0) / dw, 0, 8 - 1e-5f),
              hi = nh == 1 ? 0 : clipf((h - h0) / dh, 0, 8 - 1e-5f);
        float fw = wi - (int)wi, fh = hi - (int)hi, val = 0;
        if (deriv)
            for (int d = 0; d < 4; d++)
                deriv[d] = 0;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < min(nh, 2); b++) {
                float ww = a ? fw : 1 - fw, wh = b ? fh : 1 - fh;
                const float *p = data + ((((int)wi + a) * nh + (int)hi + b) * n + iy) * n + ix;
                float q = (1 - fy) * ((1 - fx) * p[0] + fx * p[1]) +
                          fy * ((1 - fx) * p[n] + fx * p[n + 1]);
                val += ww * wh * q;
                if (deriv) {
                    deriv[0] -= 5 * ww * wh * ((1 - fy) * (p[1] - p[0]) + fy * (p[n + 1] - p[n]));
                    deriv[1] -= 5 * ww * wh * ((1 - fx) * (p[n] - p[0]) + fx * (p[n + 1] - p[1]));
                    deriv[2] += (a ? 1 : -1) * wh * q / dw;
                    if (nh > 1)
                        deriv[3] += (b ? 1 : -1) * ww * q / dh;
                }
            }
        return val;
    }
};

__device__ float warp_sum(float v) {
    for (int s = 16; s; s /= 2)
        v += __shfl_down_sync(0xffffffff, v, s);
    return v;
}

// Cholesky solve of the scaled, damped Gram matrix. For a condition check,
// positive definiteness after subtracting N/20000 proves the ratio is safe.
__device__ bool cholesky(float a[6][6], int n, float diagonal_shift) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float v = a[i][j] + (i == j ? diagonal_shift : 0);
            for (int k = 0; k < j; k++)
                v -= a[i][k] * a[j][k];
            if (j == i) {
                if (v <= 0)
                    return false;
                a[i][i] = sqrtf(v);
            } else
                a[i][j] = v / a[j][j];
        }
    }
    return true;
}

__device__ bool well_conditioned(float corr[6][6], int n) {
    float a[6][6];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = corr[i][j];
    if (cholesky(a, n, -(n + 0.01f) / 20000.f))
        return true;
    // The sufficient bound is inconclusive near a singular solution.
    double e[6][6];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            e[i][j] = corr[i][j];
    for (int sweep = 0; sweep < 16; sweep++) {
        double largest = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double off = e[i][j];
                largest = fmax(largest, fabs(off));
                if (fabs(off) < 1e-12)
                    continue;
                double tau = (e[j][j] - e[i][i]) / (2 * off);
                double t = copysign(1., tau) / (fabs(tau) + hypot(1., tau));
                double c = rsqrt(1 + t * t), s = t * c;
                e[i][i] -= t * off;
                e[j][j] += t * off;
                e[i][j] = e[j][i] = 0;
                for (int k = 0; k < n; k++)
                    if (k != i && k != j) {
                        double p = e[i][k], q = e[j][k];
                        e[i][k] = e[k][i] = c * p - s * q;
                        e[j][k] = e[k][j] = s * p + c * q;
                    }
            }
        if (largest < 1e-10)
            break;
    }
    double lo = e[0][0], hi = lo;
    for (int i = 1; i < n; i++) {
        lo = fmin(lo, e[i][i]);
        hi = fmax(hi, e[i][i]);
    }
    return hi <= 2e4 * fmax(lo, 1e-9);
}

extern "C" __global__ void __launch_bounds__(ACTIS_FIT_THREADS)
    fit_apertures(const float *tiles, int side, const double *centers, const float *seeds,
                  const int *near, const int *owner, const int *use, int half, const float *tables,
                  const float *bounds, int radius, int nh, int steps, int profile, float *result) {
    int job = blockIdx.x, site = use[job], tile = owner[site], t = threadIdx.x;
    int stamp = 2 * half + 1, npix = stamp * stamp, n = 10 * radius + 1, dim = nh == 1 ? 5 : 6;
    // Every tile of the page shares one template library and its size range.
    const float *b = bounds;
    Aperture lib = {tables, n, nh, radius, b[0], (b[1] - b[0]) / 8, b[2], (b[3] - b[2]) / 8};
    int px = t % stamp - half, py = t / stamp - half;
    double cx = centers[2 * site], cy = centers[2 * site + 1];
    int ix = __double2int_rn(cx), iy = __double2int_rn(cy);
    float x = (float)((ix + px) - cx), y = (float)((iy + py) - cy);
    float z = 0, tail = 0;
    extern __shared__ float buf[];
    __shared__ float params[6], lo, scale, noise, partial[28][32];
    if (t < npix) {
        z = tiles[(tile * side + iy + py) * side + ix + px];
        for (int j = 0; j < 8; j++) {
            int adj = near[site * 8 + j];
            if (adj < 0)
                continue;
            double dx = centers[2 * adj] - cx, dy = centers[2 * adj + 1] - cy;
            if (fmax(fabs(dx), fabs(dy)) >= radius + half)
                continue;
            float gain = clipf(seeds[3 * adj + 2] / seeds[3 * site + 2], 0, 4);
            tail += gain *
                    lib.sample(x - (float)dx, y - (float)dy, seeds[3 * adj], seeds[3 * adj + 1], 0);
        }
    }
    buf[t] = t < npix ? z : __int_as_float(0x7f800000);
    __syncthreads();
    sort_block(buf, blockDim.x);
    if (!t) {
        lo = buf[0];
        scale = fmaxf(buf[npix - 1] - lo, 1e-8f);
        params[0] = 0;
        params[1] = 1;
        params[2] = params[3] = 0;
        params[4] = clipf(seeds[3 * site], b[0] + .001f, b[1] - .001f);
        params[5] = nh == 1 ? params[4] : clipf(seeds[3 * site + 1], b[2] + .001f, b[3] - .001f);
    }
    __syncthreads();
    z = (z - lo) / scale;
    int warp = t / 32, lane = t % 32, nwarps = blockDim.x / 32;
    for (int it = 0; it <= steps; it++) {
        float d[4], jac[6] = {1, 0, 0, 0, 0, 0};
        float val =
            t < npix ? lib.sample(x - params[2], y - params[3], params[4], params[5], d) + tail : 0;
        if (!it) {
            float ts = warp_sum(val), zs = warp_sum(t < npix ? z : 0);
            if (!lane) {
                partial[0][warp] = ts;
                partial[1][warp] = zs;
            }
            __syncthreads();
            if (!t) {
                float mt = 0, mz = 0;
                for (int j = 0; j < nwarps; j++) {
                    mt += partial[0][j];
                    mz += partial[1][j];
                }
                buf[0] = mt / npix;
                buf[1] = mz / npix;
            }
            __syncthreads();
            float mt = buf[0], mz = buf[1], dt = val - mt;
            float tz = warp_sum(t < npix ? dt * z : 0), tt = warp_sum(t < npix ? dt * dt : 0);
            if (!lane) {
                partial[0][warp] = tz;
                partial[1][warp] = tt;
            }
            __syncthreads();
            if (!t) {
                float a = 0, c = 0;
                for (int j = 0; j < nwarps; j++) {
                    a += partial[0][j];
                    c += partial[1][j];
                }
                params[1] = clipf(a / fmaxf(c, 1e-7f), .05f, 12.f);
                params[0] = mz - params[1] * mt;
            }
            __syncthreads();
        }
        float res = t < npix ? z - params[0] - params[1] * val : 0;
        if (it == steps) {
            float ss = warp_sum(res * res);
            if (!lane)
                partial[0][warp] = ss;
            __syncthreads();
            if (!t) {
                float sum = 0;
                for (int j = 0; j < nwarps; j++)
                    sum += partial[0][j];
                result[2 * job] = sqrtf(sum / npix);
                result[2 * job + 1] = params[4];
            }
            __syncthreads();
            if (profile)
                return;
        }
        buf[t] = t < npix ? fabsf(res) : __int_as_float(0x7f800000);
        __syncthreads();
        sort_block(buf, blockDim.x);
        if (!t)
            noise = fmaxf(.015f, 1.4826f * buf[npix / 2]);
        __syncthreads();
        float wt = t < npix ? fminf(1, 2 * noise / fmaxf(fabsf(res), 1e-7f)) : 0;
        jac[1] = val;
        for (int j = 2; j < dim; j++)
            jac[j] = t < npix ? params[1] * d[j - 2] : 0;
        int k = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j <= i; j++, k++) {
                float g = warp_sum(wt * jac[i] * jac[j]);
                if (!lane)
                    partial[k][warp] = g;
            }
            float v = warp_sum(wt * jac[i] * res);
            if (!lane)
                partial[k][warp] = v;
            k++;
        }
        __syncthreads();
        if (t < k) {
            float sum = 0;
            for (int j = 0; j < nwarps; j++)
                sum += partial[t][j];
            buf[t] = sum;
        }
        __syncthreads();
        if (!t) {
            float a[6][6], norm[6], rhs[6];
            int k = 0;
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j <= i; j++)
                    a[i][j] = a[j][i] = buf[k++];
                norm[i] = sqrtf(fmaxf(a[i][i], 1e-7f));
                rhs[i] = buf[k++] / norm[i];
            }
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    a[i][j] /= norm[i] * norm[j];
            if (it == steps) {
                bool valid = well_conditioned(a, dim) && result[2 * job] <= .16f &&
                             fabsf(params[2]) <= 1.98f && fabsf(params[3]) <= 1.98f &&
                             params[4] >= b[0] + .02f && params[4] <= b[1] - .02f &&
                             (nh == 1 || (params[5] >= b[2] + .02f && params[5] <= b[3] - .02f));
                result[2 * job + 1] = valid ? fmaxf(params[4], params[5]) : nanf("");
            } else {
                cholesky(a, dim, 3e-4f);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < i; j++)
                        rhs[i] -= a[i][j] * rhs[j];
                    rhs[i] /= a[i][i];
                }
                for (int i = dim - 1; i >= 0; i--) {
                    for (int j = i + 1; j < dim; j++)
                        rhs[i] -= a[j][i] * rhs[j];
                    rhs[i] /= a[i][i];
                }
                for (int i = 0; i < dim; i++)
                    rhs[i] /= norm[i];
                params[0] += clipf(rhs[0], -.5f, .5f);
                params[1] = clipf(params[1] + clipf(rhs[1], -1.f, 1.f), .02f, 12.f);
                for (int i = 2; i <= 3; i++)
                    params[i] = clipf(params[i] + clipf(rhs[i], -.8f, .8f), -2, 2);
                params[4] = clipf(params[4] + clipf(rhs[4], -1.8f, 1.8f), b[0], b[1]);
                params[5] =
                    nh == 1 ? params[4] : clipf(params[5] + clipf(rhs[5], -1.8f, 1.8f), b[2], b[3]);
            }
        }
        __syncthreads();
    }
}
