"""Fused sampling of the aperture library for the batched CUDA solver."""

from functools import lru_cache


@lru_cache(maxsize=2)
def aperture_sampler(derivatives):
    import cupy as cp

    outputs = "float32 value"
    if derivatives:
        outputs += ", float32 dx, float32 dy, float32 dw, float32 dh"
    return cp.ElementwiseKernel(
        "T x, T y, W width, H height, raw float32 library, "
        "int32 side, int32 nw, int32 nh, float32 radius, float32 scale, "
        "float32 w0, float32 h0, float32 wstep, float32 hstep",
        outputs,
        r"""
        const float cx = fminf(fmaxf((x + radius) * scale, 0), side - 1);
        const float cy = fminf(fmaxf((y + radius) * scale, 0), side - 1);
        const int ix = min((int)cx, side - 2), iy = min((int)cy, side - 2);
        const float fx = cx - ix, fy = cy - iy;
        const float wi = fminf(fmaxf((width - w0) / wstep, 0), nw - 1 - 1e-5f);
        const float hi = nh == 1 ? 0 :
            fminf(fmaxf((height - h0) / hstep, 0), nh - 1 - 1e-5f);
        const float fw = wi - (int)wi, fh = hi - (int)hi;
        value = 0;
        #if ACTIS_GRADIENT
        dx = dy = dw = dh = 0;
        #endif
        for (int iw = 0; iw < 2; ++iw) {
            const float ww = iw ? fw : 1 - fw;
            for (int ih = 0; ih < min(nh, 2); ++ih) {
                const float wh = ih ? fh : 1 - fh;
                const int plane = ((int)wi + iw) * nh + (int)hi + ih;
                const int offset = (plane * side + iy) * side + ix;
                const float a = library[offset], b = library[offset + 1];
                const float c = library[offset + side], d = library[offset + side + 1];
                const float sample = (1-fy)*((1-fx)*a + fx*b) + fy*((1-fx)*c + fx*d);
                value += ww * wh * sample;
                #if ACTIS_GRADIENT
                dx -= scale * ww * wh * ((1-fy)*(b-a) + fy*(d-c));
                dy -= scale * ww * wh * ((1-fx)*(c-a) + fx*(d-b));
                dw += (iw ? 1 : -1) * wh * sample / wstep;
                if (nh > 1) dh += (ih ? 1 : -1) * ww * sample / hstep;
                #endif
            }
        }
        """,
        name="actis_aperture_gradient" if derivatives else "actis_aperture_value",
        options=(f"-DACTIS_GRADIENT={int(derivatives)}",),
    )
