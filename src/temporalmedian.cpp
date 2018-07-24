#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <VapourSynth.h>
#include <VSHelper.h>


enum {
    MaxRadius = 10,
    MaxDiameter = MaxRadius * 2 + 1
};


template <typename PixelType>
static void process_plane_scalar(
        const uint8_t *srcp8[MaxDiameter],
        uint8_t *dstp8,
        int width,
        int height,
        int stride,
        int diameter) {

    const PixelType *srcp[MaxDiameter];

    for (int i = 0; i < MaxDiameter; i++)
        srcp[i] = (const PixelType *)srcp8[i];

    PixelType *dstp = (PixelType *)dstp8;
    stride /= sizeof(PixelType);

    PixelType temp[MaxDiameter];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int i = 0; i < diameter; i++)
                temp[i] = srcp[i][x];

            std::sort(&temp[0], &temp[diameter]);

            dstp[x] = temp[diameter >> 1];
        }

        for (int i = 0; i < diameter; i++)
            srcp[i] += stride;
        dstp += stride;
    }
}


#if defined (TEMPORAL_MEDIAN_X86)

#include <emmintrin.h>


#ifdef _WIN32
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif


template <typename PixelType, bool aligned>
static FORCE_INLINE __m128i mm_load_16(const uint8_t *srcp) {
    if (sizeof(PixelType) == 4) {
        __m128 result = aligned ? _mm_load_ps((const float *)srcp)
                                : _mm_loadu_ps((const float *)srcp);
        return _mm_castps_si128(result);
    } else {
        __m128i result = aligned ? _mm_load_si128((const __m128i *)srcp)
                                 : _mm_loadu_si128((const __m128i *)srcp);

        // For pminsw/pmaxsw.
        if (sizeof(PixelType) == 2)
            result = _mm_add_epi16(result, _mm_set1_epi16(0x8000));

        return result;
    }
}


template <typename PixelType, bool aligned>
static FORCE_INLINE void mm_store_16(uint8_t *srcp, const __m128i &value) {
    if (sizeof(PixelType) == 4) {
        aligned ? _mm_store_ps((float *)srcp, _mm_castsi128_ps(value))
                : _mm_storeu_ps((float *)srcp, _mm_castsi128_ps(value));
    } else {
        __m128i result = value;

        // For pminsw/pmaxsw.
        if (sizeof(PixelType) == 2)
            result = _mm_add_epi16(result, _mm_set1_epi16(0x8000));

        aligned ? _mm_store_si128((__m128i *)srcp, result)
                : _mm_storeu_si128((__m128i *)srcp, result);
    }
}


template <typename PixelType>
static FORCE_INLINE __m128i mm_min(const __m128i &a, const __m128i &b) {
    if (sizeof(PixelType) == 1)
        return _mm_min_epu8(a, b);
    else if (sizeof(PixelType) == 2)
        return _mm_min_epi16(a, b);
    else
        return _mm_castps_si128(_mm_min_ps(_mm_castsi128_ps(a),
                                           _mm_castsi128_ps(b)));
}


template <typename PixelType>
static FORCE_INLINE __m128i mm_max(const __m128i &a, const __m128i &b) {
    if (sizeof(PixelType) == 1)
        return _mm_max_epu8(a, b);
    else if (sizeof(PixelType) == 2)
        return _mm_max_epi16(a, b);
    else
        return _mm_castps_si128(_mm_max_ps(_mm_castsi128_ps(a),
                                           _mm_castsi128_ps(b)));
}


template <typename PixelType>
static FORCE_INLINE __m128i mm_median3(const __m128i &a, const __m128i &b, const __m128i &c) {
    return mm_max<PixelType>(mm_min<PixelType>(a, b),
                             mm_min<PixelType>(c,
                                               mm_max<PixelType>(a, b)));
}


template <typename PixelType, bool aligned>
static FORCE_INLINE void process_xmmword_sse2(const uint8_t *srcp[MaxDiameter], uint8_t *dstp, int x, int radius) {
    __m128i a = mm_load_16<PixelType, aligned>(&srcp[0][x]);
    __m128i b = mm_load_16<PixelType, aligned>(&srcp[1][x]);
    __m128i c = mm_load_16<PixelType, aligned>(&srcp[2][x]);

    if (radius == 2) {
        __m128i d = mm_load_16<PixelType, aligned>(&srcp[3][x]);
        __m128i e = mm_load_16<PixelType, aligned>(&srcp[4][x]);

        __m128i f = mm_max<PixelType>(mm_min<PixelType>(a, b),
                                      mm_min<PixelType>(c, d));
        __m128i g = mm_min<PixelType>(mm_max<PixelType>(a, b),
                                      mm_max<PixelType>(c, d));
        a = e;
        b = f;
        c = g;
    }

    __m128i median = mm_median3<PixelType>(a, b, c);

    mm_store_16<PixelType, aligned>(&dstp[x], median);
}


template <typename PixelType, int radius>
static void process_plane_sse2(
        const uint8_t *srcp[MaxDiameter],
        uint8_t *dstp,
        int width,
        int height,
        int stride,
        int diameter) {
    (void)diameter;

    int row_size = width * sizeof(PixelType);
    int row_size_simd = row_size / 16 * 16;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < row_size_simd; x += 16)
            process_xmmword_sse2<PixelType, true>(srcp, dstp, x, radius);

        if (row_size_simd < row_size)
            process_xmmword_sse2<PixelType, false>(srcp, dstp, row_size - 16, radius);

        for (int i = 0; i < radius * 2 + 1; i++)
            srcp[i] += stride;
        dstp += stride;
    }
}

#endif


typedef struct TemporalMedianData {
    VSNodeRef *clip;
    const VSVideoInfo *vi;

    int radius;
    int process[3];

    decltype (process_plane_scalar<uint8_t>) *process_plane;
} TemporalMedianData;


static void VS_CC temporalMedianInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    TemporalMedianData *d = (TemporalMedianData *) *instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC temporalMedianGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const TemporalMedianData *d = (const TemporalMedianData *) *instanceData;

    if (activationReason == arInitial) {
        if (n < d->radius || n > d->vi->numFrames - 1 - d->radius) {
            vsapi->requestFrameFilter(n, d->clip, frameCtx);
        } else {
            for (int i = -d->radius; i <= d->radius; i++)
                vsapi->requestFrameFilter(n + i, d->clip, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        if (n < d->radius || n > d->vi->numFrames - 1 - d->radius)
            return vsapi->getFrameFilter(n, d->clip, frameCtx);

        int diameter = d->radius * 2 + 1;

        const VSFrameRef *src_frames[MaxDiameter];
        const uint8_t *srcp[MaxDiameter];

        for (int i = -d->radius; i <= d->radius; i++)
            src_frames[d->radius + i] = vsapi->getFrameFilter(n + i, d->clip, frameCtx);

        const VSFrameRef *plane_src[3] = {
            d->process[0] ? nullptr : src_frames[d->radius],
            d->process[1] ? nullptr : src_frames[d->radius],
            d->process[2] ? nullptr : src_frames[d->radius]
        };
        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, plane_src, planes, src_frames[d->radius], core);


        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!d->process[plane])
                continue;

            int width = vsapi->getFrameWidth(dst, plane);
            int height = vsapi->getFrameHeight(dst, plane);
            int stride = vsapi->getStride(dst, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);

            for (int i = 0; i < diameter; i++)
                srcp[i] = vsapi->getReadPtr(src_frames[i], plane);

            d->process_plane(srcp, dstp, width, height, stride, diameter);
        }


        for (int i = -d->radius; i <= d->radius; i++)
            vsapi->freeFrame(src_frames[d->radius + i]);

        return dst;
    }

    return nullptr;
}


static void VS_CC temporalMedianFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    TemporalMedianData *d = (TemporalMedianData *)instanceData;

    vsapi->freeNode(d->clip);
    free(d);
}


static void VS_CC temporalMedianCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    TemporalMedianData d;
    memset(&d, 0, sizeof(d));

    int err;

    d.radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &err));
    if (err)
        d.radius = 1;


    if (d.radius < 1 || d.radius > 10) {
        vsapi->setError(out, "TemporalMedian: radius must be between 1 and 10 (inclusive).");
        return;
    }


    d.clip = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.clip);


    if (!d.vi->format ||
        (d.vi->format->sampleType == stInteger && d.vi->format->bitsPerSample > 16) ||
        (d.vi->format->sampleType == stFloat && d.vi->format->bitsPerSample != 32) ||
        d.vi->width == 0 ||
        d.vi->height == 0) {
        vsapi->setError(out, "TemporalMedian: clip must be 8..16 bit integer or 32 bit float with constant format and dimensions.");
        vsapi->freeNode(d.clip);
        return;
    }


    int n = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, 0));

        if (o < 0 || o >= n) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "TemporalMedian: plane index out of range");
            return;
        }

        if (d.process[o]) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "TemporalMedian: plane specified twice");
            return;
        }

        d.process[o] = 1;
    }


    if (d.vi->format->bitsPerSample == 8) {
        d.process_plane = process_plane_scalar<uint8_t>;
    } else if (d.vi->format->sampleType == stInteger) {
        d.process_plane = process_plane_scalar<uint16_t>;
    } else {
        d.process_plane = process_plane_scalar<float>;
    }

#if defined (TEMPORAL_MEDIAN_X86)
    if (d.radius < 3) {
        if (d.vi->format->bitsPerSample == 8) {
            d.process_plane = (d.radius == 1) ? process_plane_sse2<uint8_t, 1>
                                              : process_plane_sse2<uint8_t, 2>;
        } else if (d.vi->format->sampleType == stInteger) {
            d.process_plane = (d.radius == 1) ? process_plane_sse2<uint16_t, 1>
                                              : process_plane_sse2<uint16_t, 2>;
        } else {
            d.process_plane = (d.radius == 1) ? process_plane_sse2<float, 1>
                                              : process_plane_sse2<float, 2>;
        }
    }
#endif


    TemporalMedianData *data = (TemporalMedianData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "TemporalMedian", temporalMedianInit, temporalMedianGetFrame, temporalMedianFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.temporalmedian", "tmedian", "Calculates temporal median", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("TemporalMedian",
                 "clip:clip;"
                 "radius:int:opt;"
                 "planes:int[]:opt;"
                 , temporalMedianCreate, 0, plugin);
}
