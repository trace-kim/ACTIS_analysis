/* Bounded Levenberg-Marquardt fit of signed lattice harmonics, in double precision.
   One thread owns one start of one tile. The model and its analytic derivatives
   are those of harmonics._fit_orders. */
#define PI 3.141592653589793

__device__ double sinc_pi(double x) { return x == 0 ? 1.0 : sin(PI * x) / (PI * x); }

struct Orders {
    const double *freq, *target;
    const unsigned char *use;
    int count, shape, dim;
    double error, pitch, cos_a, sin_a;

    // Half the squared residual, its gradient and the Gauss-Newton matrix at z.
    __device__ double evaluate(const double *z, double *g, double G[6][6]) const {
        double wid = z[0] * pitch, ht = (shape == 2 ? z[1] : z[0]) * pitch;
        double sx = z[dim - 3] * pitch, sy = z[dim - 2] * pitch, gain = exp(z[dim - 1]);
        double cost = 0;
        for (int i = 0; i < dim; i++) {
            g[i] = 0;
            for (int j = 0; j < dim; j++)
                G[i][j] = 0;
        }
        for (int k = 0; k < count; k++) {
            if (!use[k])
                continue;
            double fx = freq[2 * k], fy = freq[2 * k + 1];
            double along = fx * cos_a + fy * sin_a, across = -fx * sin_a + fy * cos_a;
            double u = wid * along, v = ht * across, form, dw, dh;
            if (shape == 1) {
                double su = sinc_pi(u), sv = sinc_pi(v);
                form = su * sv;
                dw = (cos(PI * u) - su) * sv / wid;
                dh = (cos(PI * v) - sv) * su / ht;
            } else {
                double r = PI * hypot(u, v);
                form = r > 1e-8 ? 2 * j1(r) / r : 1.0;
                double slope = (r > 1e-4 ? 2 * (j0(r) - form) / (r * r) : -0.25) * PI * PI;
                dw = slope * wid * along * along;
                dh = slope * ht * across * across;
            }
            double mtf = exp(-2 * PI * PI * (sx * fx * sx * fx + sy * fy * sy * fy));
            double scale = gain * sinc_pi(fx) * sinc_pi(fy) * mtf / error;
            double residual = scale * form - target[k] / error;
            double blur = -4 * PI * PI * pitch * form, row[6];
            int n = 0;
            if (shape == 2) {
                row[n++] = pitch * dw;
                row[n++] = pitch * dh;
            } else
                row[n++] = pitch * (dw + dh);
            row[n++] = blur * sx * fx * fx;
            row[n++] = blur * sy * fy * fy;
            row[n++] = form;
            cost += residual * residual;
            for (int i = 0; i < dim; i++) {
                row[i] *= scale;
                g[i] += row[i] * residual;
                for (int j = 0; j <= i; j++)
                    G[i][j] += row[i] * row[j];
            }
        }
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < i; j++)
                G[j][i] = G[i][j];
        return 0.5 * cost;
    }
};

// Solve a x = b in place by Cholesky. False if a is not positive definite.
__device__ bool solve_spd(double a[6][6], double *b, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            double v = a[i][j];
            for (int k = 0; k < j; k++)
                v -= a[i][k] * a[j][k];
            if (j < i)
                a[i][j] = v / a[j][j];
            else if (v > 0)
                a[i][i] = sqrt(v);
            else
                return false;
        }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++)
            b[i] -= a[i][j] * b[j];
        b[i] /= a[i][i];
    }
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++)
            b[i] -= a[j][i] * b[j];
        b[i] /= a[i][i];
    }
    return true;
}

/* state: 0 evaluation limit, 1 gradient, 2 cost, 3 step. Any other than 0 has converged. */
extern "C" __global__ void fit_harmonics(const double *freq, const double *target,
                                         const unsigned char *use, const double *error,
                                         const double *starts, int orders, int tiles, int n_starts,
                                         double pitch, double cos_a, double sin_a, int shape,
                                         int max_nfev, double tol, double *solution, double *cost,
                                         int *state_out, double *gram) {
    int job = blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= tiles * n_starts)
        return;
    int tile = job / n_starts, dim = shape == 2 ? 5 : 4;
    Orders model = {freq,       target + (long long)tile * orders, use + (long long)tile * orders,
                    orders,     shape,                             dim,
                    error[tile], pitch,                            cos_a,
                    sin_a};
    double lo[6], hi[6], z[6], width = starts[job % n_starts];
    int n = 0;
    lo[n] = .075, hi[n] = 1.2, z[n++] = width;
    if (shape == 2)
        lo[n] = .075, hi[n] = 1.2, z[n++] = .78 * width;
    lo[n] = .004, hi[n] = .35, z[n++] = .065;
    lo[n] = .004, hi[n] = .35, z[n++] = .065;
    lo[n] = -2, hi[n] = 4, z[n++] = .25;
    double g[6], G[6][6], F = model.evaluate(z, g, G);
    double damping = 1e-3, growth = 2;
    int nfev = 1, state = 0;
    while (!state && nfev < max_nfev) {
        // A parameter resting on a bound with the cost falling beyond it stays there.
        bool held[6];
        double slope = 0, top = 0;
        for (int i = 0; i < dim; i++) {
            held[i] = (z[i] <= lo[i] && g[i] > 0) || (z[i] >= hi[i] && g[i] < 0);
            if (!held[i])
                slope = fmax(slope, fabs(g[i]));
            top = fmax(top, G[i][i]);
        }
        if (slope < tol) {
            state = 1;
            break;
        }
        double a[6][6], step[6];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++)
                a[i][j] = held[i] || held[j] ? (i == j) : G[i][j];
            if (!held[i])
                a[i][i] += damping * fmax(G[i][i], 1e-12 * top) + 1e-300;
            step[i] = held[i] ? 0 : -g[i];
        }
        if (!solve_spd(a, step, dim)) {
            damping *= 10;
            if (damping > 1e30)
                break;
            continue;
        }
        double trial[6], moved = 0, size = 0;
        for (int i = 0; i < dim; i++) {
            trial[i] = fmin(hi[i], fmax(lo[i], z[i] + step[i]));
            step[i] = trial[i] - z[i];
            moved += step[i] * step[i];
            size += z[i] * z[i];
        }
        if (sqrt(moved) < tol * (tol + sqrt(size))) {
            state = 3;
            break;
        }
        double gt[6], Gt[6][6], Ft = model.evaluate(trial, gt, Gt), predicted = 0;
        nfev++;
        for (int i = 0; i < dim; i++) {
            double curved = 0;
            for (int j = 0; j < dim; j++)
                curved += G[i][j] * step[j];
            predicted -= step[i] * (g[i] + .5 * curved);
        }
        double ratio = predicted > 0 ? (F - Ft) / predicted : -1;
        if (Ft < F) {
            if (F - Ft < tol * F && ratio > .25)
                state = 2;
            F = Ft;
            for (int i = 0; i < dim; i++) {
                z[i] = trial[i];
                g[i] = gt[i];
                for (int j = 0; j < dim; j++)
                    G[i][j] = Gt[i][j];
            }
            double q = 2 * ratio - 1;
            damping *= fmax(1. / 3, 1 - q * q * q);
            growth = 2;
        } else {
            damping *= growth;
            growth *= 2;
        }
    }
    cost[job] = F;
    state_out[job] = state;
    for (int i = 0; i < 6; i++) {
        solution[6 * job + i] = i < dim ? z[i] : 0;
        for (int j = 0; j < 6; j++)
            gram[36 * job + 6 * i + j] = i < dim && j < dim ? G[i][j] : 0;
    }
}
