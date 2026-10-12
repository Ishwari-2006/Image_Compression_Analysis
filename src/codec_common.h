// ============================================================================
//  codec_common.h  -  shared pieces of the single-image JPEG-style codec
//
//  The codec is a small, self-written baseline-JPEG-like pipeline:
//      RGB -> YCbCr -> 8x8 DCT -> quantization -> Huffman coding -> bitstream
//  (4:4:4, i.e. no chroma subsampling).
//
//  We write it ourselves (instead of calling libjpeg-turbo) because the whole
//  point of this experiment is to look INSIDE the compression of ONE image and
//  find which stages are independent and can run in parallel.
//
//  This header contains everything that is identical for the sequential and the
//  parallel program: data structures, per-block math, Huffman, decoder,
//  metrics.  The two programs differ only in HOW the stage loops are executed.
// ============================================================================
#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // MinGW with -std=c++17 does not define M_PI
#endif

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
const int QUALITY = 75;   // same fixed JPEG quality used in the rest of the project

inline double now_sec()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Basic data structures
// ---------------------------------------------------------------------------
struct Image                       // interleaved 8-bit RGB
{
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
};

struct Geometry                    // image size padded up to multiples of 8
{
    int w, h, pw, ph, bw, bh, nb;
    Geometry(int w_, int h_) : w(w_), h(h_)
    {
        pw = (w + 7) / 8 * 8;
        ph = (h + 7) / 8 * 8;
        bw = pw / 8;
        bh = ph / 8;
        nb = bw * bh;              // number of 8x8 blocks per colour component
    }
};

struct Planes                      // level-shifted Y, Cb, Cr planes (float)
{
    std::vector<float> p[3];
    explicit Planes(const Geometry& g)
    {
        for (auto& v : p) v.assign((size_t)g.pw * g.ph, 0.0f);
    }
};

struct Coefs                       // quantized DCT coefficients, 64 per block
{
    std::vector<int16_t> c[3];
    explicit Coefs(const Geometry& g)
    {
        for (auto& v : c) v.assign((size_t)g.nb * 64, 0);
    }
};

struct Timings                     // wall-clock seconds of every compression stage
{
    double color = 0, dct = 0, hist = 0, huff = 0, enc = 0, merge = 0, total = 0;
    void add(const Timings& o)
    {
        color += o.color; dct += o.dct; hist += o.hist; huff += o.huff;
        enc += o.enc; merge += o.merge; total += o.total;
    }
    void scale(double f)
    {
        color *= f; dct *= f; hist *= f; huff *= f; enc *= f; merge *= f; total *= f;
    }
};

struct DTimings                    // wall-clock seconds of every decompression stage
{
    double parse = 0, entropy = 0, idct = 0, color = 0, total = 0;
    void add(const DTimings& o)
    {
        parse += o.parse; entropy += o.entropy; idct += o.idct; color += o.color; total += o.total;
    }
    void scale(double f) { parse *= f; entropy *= f; idct *= f; color *= f; total *= f; }
};

// The bit stream is cut into chunks of CHUNK_BLOCKS blocks (same for every run, thread
// count and program).  The file header stores where every chunk starts, so chunks can
// be decoded independently.
const int CHUNK_BLOCKS = 512;

struct ChunkInfo
{
    uint64_t bitOffset;            // where the chunk starts inside the payload
    int16_t  dc[3];                // DC value of the block before the chunk (DC prediction)
};

// ---------------------------------------------------------------------------
// Constant tables
// ---------------------------------------------------------------------------
static const int ZZ[64] = {
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63 };

static const int BASE_Q_LUMA[64] = {
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99 };

static const int BASE_Q_CHROMA[64] = {
    17,18,24,47,99,99,99,99, 18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99, 47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99 };

static float DCTM[8][8];           // DCT-II basis matrix
static int   QT[2][64];            // quantization tables: [0]=luma, [1]=chroma

inline void init_codec(int quality)
{
    for (int u = 0; u < 8; ++u)
        for (int x = 0; x < 8; ++x)
        {
            double a = (u == 0) ? std::sqrt(1.0 / 8.0) : std::sqrt(2.0 / 8.0);
            DCTM[u][x] = (float)(a * std::cos((2 * x + 1) * u * M_PI / 16.0));
        }
    int scale = (quality < 50) ? 5000 / quality : 200 - 2 * quality;
    for (int i = 0; i < 64; ++i)
    {
        QT[0][i] = std::clamp((BASE_Q_LUMA[i]   * scale + 50) / 100, 1, 255);
        QT[1][i] = std::clamp((BASE_Q_CHROMA[i] * scale + 50) / 100, 1, 255);
    }
}

// ---------------------------------------------------------------------------
// Image I/O
// ---------------------------------------------------------------------------
inline bool load_image(const std::string& path, Image& im)
{
    int n = 0;
    uint8_t* d = stbi_load(path.c_str(), &im.w, &im.h, &n, 3);
    if (!d) return false;
    im.rgb.assign(d, d + (size_t)im.w * im.h * 3);
    stbi_image_free(d);
    return true;
}

inline void save_png(const std::string& path, const Image& im)
{
    stbi_write_png(path.c_str(), im.w, im.h, 3, im.rgb.data(), im.w * 3);
}

// ===========================================================================
//  STAGE 1  -  colour conversion RGB -> YCbCr  (one image row at a time)
//  Every output pixel depends only on the SAME input pixel  => fully independent
// ===========================================================================
inline void color_row(const Image& im, const Geometry& g, Planes& P, int y)
{
    int sy = std::min(y, g.h - 1);                       // edge replication
    const uint8_t* src = &im.rgb[(size_t)sy * g.w * 3];
    for (int x = 0; x < g.pw; ++x)
    {
        int sx = std::min(x, g.w - 1);
        float r = src[sx * 3], gg = src[sx * 3 + 1], b = src[sx * 3 + 2];
        size_t o = (size_t)y * g.pw + x;
        P.p[0][o] =  0.299f    * r + 0.587f    * gg + 0.114f    * b - 128.0f;
        P.p[1][o] = -0.168736f * r - 0.331264f * gg + 0.5f      * b;
        P.p[2][o] =  0.5f      * r - 0.418688f * gg - 0.081312f * b;
    }
}

// ===========================================================================
//  STAGE 2  -  forward DCT + quantization of ONE 8x8 block of ONE component
//  A block only reads its own 64 pixels and only writes its own 64
//  coefficients  => every block (and every component) is independent
// ===========================================================================
inline void dct_quant_block(const Planes& P, const Geometry& g, int c, int b,
                            int16_t* out)
{
    int bx = b % g.bw, by = b / g.bw;
    const float* base = P.p[c].data() + (size_t)(by * 8) * g.pw + bx * 8;
    const int*   q    = QT[c > 0 ? 1 : 0];
    float tmp[64];

    for (int u = 0; u < 8; ++u)                           // tmp = C * B
        for (int x = 0; x < 8; ++x)
        {
            float s = 0;
            for (int y = 0; y < 8; ++y) s += DCTM[u][y] * base[(size_t)y * g.pw + x];
            tmp[u * 8 + x] = s;
        }
    for (int u = 0; u < 8; ++u)                           // F = tmp * C^T, then quantize
        for (int v = 0; v < 8; ++v)
        {
            float s = 0;
            for (int x = 0; x < 8; ++x) s += tmp[u * 8 + x] * DCTM[v][x];
            out[u * 8 + v] = (int16_t)lrintf(s / (float)q[u * 8 + v]);
        }
}

// ---------------------------------------------------------------------------
// Entropy-coding symbol generation (shared by "count" and "write" passes)
// ---------------------------------------------------------------------------
inline int bitlen(int v) { int n = 0; while (v) { ++n; v >>= 1; } return n; }

// Table index: 0 = luma DC, 1 = luma AC, 2 = chroma DC, 3 = chroma AC
template <class Sink>
inline void walk_block(Sink& s, int comp, const int16_t* blk, int prevDC)
{
    int cls = comp > 0 ? 2 : 0;
    int d = blk[0] - prevDC;                              // DC is coded as a difference
    int sz = bitlen(std::abs(d));
    s.sym(cls, sz);
    if (sz) s.bits(d > 0 ? d : d + (1 << sz) - 1, sz);

    int run = 0;
    for (int k = 1; k < 64; ++k)                          // AC in zig-zag order
    {
        int v = blk[ZZ[k]];
        if (v == 0) { ++run; continue; }
        while (run > 15) { s.sym(cls + 1, 0xF0); run -= 16; }
        int z = bitlen(std::abs(v));
        s.sym(cls + 1, (run << 4) | z);
        s.bits(v > 0 ? v : v + (1 << z) - 1, z);
        run = 0;
    }
    if (run > 0) s.sym(cls + 1, 0x00);                    // end-of-block
}

struct Hist
{
    uint64_t n[4][256];
    Hist() { std::memset(n, 0, sizeof(n)); }
    void add(const Hist& o)
    {
        for (int t = 0; t < 4; ++t) for (int s = 0; s < 256; ++s) n[t][s] += o.n[t][s];
    }
};

struct HistSink
{
    Hist* h;
    void sym(int t, int s)    { h->n[t][s]++; }
    void bits(int, int)       {}
};

// ---------------------------------------------------------------------------
// Bit writer
// ---------------------------------------------------------------------------
struct BitWriter
{
    std::vector<uint8_t> buf;
    uint64_t acc = 0;
    int      nacc = 0;
    uint64_t bits = 0;                                    // total bits written

    void put(uint64_t code, int len)
    {
        if (len == 0) return;
        acc = (acc << len) | (code & ((len == 64) ? ~0ULL : ((1ULL << len) - 1)));
        nacc += len; bits += len;
        while (nacc >= 8)
        {
            buf.push_back((uint8_t)((acc >> (nacc - 8)) & 0xFF));
            nacc -= 8;
        }
        acc &= (nacc == 0) ? 0 : ((1ULL << nacc) - 1);
    }
    void finish()                                         // pad last byte with zeros
    {
        if (nacc > 0) { buf.push_back((uint8_t)((acc << (8 - nacc)) & 0xFF)); nacc = 0; acc = 0; }
    }
    // append another (finished) bit stream, bit-exactly
    void append(const BitWriter& o)
    {
        uint64_t full = o.bits / 8; int rem = (int)(o.bits % 8);
        for (uint64_t i = 0; i < full; ++i) put(o.buf[i], 8);
        if (rem) put(o.buf[full] >> (8 - rem), rem);
    }
};

// ---------------------------------------------------------------------------
// Huffman coding (canonical codes)
// ---------------------------------------------------------------------------
struct HuffTable
{
    uint8_t  len[256];
    uint64_t code[256];
    HuffTable() { std::memset(len, 0, sizeof(len)); std::memset(code, 0, sizeof(code)); }
};
struct Tables { HuffTable t[4]; };

inline void assign_canonical(HuffTable& t)
{
    std::vector<int> syms;
    for (int s = 0; s < 256; ++s) if (t.len[s]) syms.push_back(s);
    std::sort(syms.begin(), syms.end(), [&](int a, int b)
              { return t.len[a] != t.len[b] ? t.len[a] < t.len[b] : a < b; });
    uint64_t code = 0; int prev = 0;
    for (int s : syms)
    {
        code <<= (t.len[s] - prev);
        t.code[s] = code++;
        prev = t.len[s];
    }
}

inline void build_huffman(const uint64_t cnt[256], HuffTable& out)
{
    struct Node { uint64_t w; int id; int l, r; };
    std::vector<Node> nodes;
    auto cmp = [&](int a, int b)                          // min-heap, deterministic ties
    {
        if (nodes[a].w != nodes[b].w) return nodes[a].w > nodes[b].w;
        return nodes[a].id > nodes[b].id;
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

    std::vector<int> leafSym;
    for (int s = 0; s < 256; ++s)
        if (cnt[s]) { nodes.push_back({cnt[s], (int)nodes.size(), -1, -1}); leafSym.push_back(s); }

    out = HuffTable();
    if (nodes.empty()) return;
    if (nodes.size() == 1) { out.len[leafSym[0]] = 1; assign_canonical(out); return; }

    int nleaf = (int)nodes.size();
    for (int i = 0; i < nleaf; ++i) pq.push(i);
    while (pq.size() > 1)
    {
        int a = pq.top(); pq.pop();
        int b = pq.top(); pq.pop();
        nodes.push_back({nodes[a].w + nodes[b].w, (int)nodes.size(), a, b});
        pq.push((int)nodes.size() - 1);
    }
    // depth of every leaf
    std::vector<std::pair<int,int>> st; st.push_back({(int)nodes.size() - 1, 0});
    while (!st.empty())
    {
        auto [n, d] = st.back(); st.pop_back();
        if (nodes[n].l < 0) { out.len[leafSym[n]] = (uint8_t)d; continue; }
        st.push_back({nodes[n].l, d + 1});
        st.push_back({nodes[n].r, d + 1});
    }
    assign_canonical(out);
}

struct EncSink
{
    BitWriter* bw;
    const Tables* T;
    void sym(int t, int s)  { bw->put(T->t[t].code[s], T->t[t].len[s]); }
    void bits(int v, int n) { bw->put((uint64_t)v, n); }
};

// ---------------------------------------------------------------------------
// File format:
//   "IMGC" | w | h | quality | 4 Huffman tables | chunk table | payload bits | payload
// ---------------------------------------------------------------------------
inline void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back((v >> (8 * i)) & 0xFF); }
inline void put64(std::vector<uint8_t>& o, uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back((v >> (8 * i)) & 0xFF); }

inline std::vector<uint8_t> make_file(const Geometry& g, const Tables& T, const BitWriter& payload,
                                      const std::vector<ChunkInfo>& chunks)
{
    std::vector<uint8_t> f;
    f.insert(f.end(), {'I', 'M', 'G', 'C'});
    put32(f, g.w); put32(f, g.h); f.push_back((uint8_t)QUALITY);
    for (int t = 0; t < 4; ++t)
    {
        int n = 0;
        for (int s = 0; s < 256; ++s) if (T.t[t].len[s]) ++n;
        f.push_back((uint8_t)(n & 0xFF)); f.push_back((uint8_t)(n >> 8));
        for (int s = 0; s < 256; ++s)
            if (T.t[t].len[s]) { f.push_back((uint8_t)s); f.push_back(T.t[t].len[s]); }
    }
    put32(f, (uint32_t)chunks.size());
    for (const ChunkInfo& c : chunks)
    {
        put64(f, c.bitOffset);
        for (int k = 0; k < 3; ++k) { f.push_back((uint8_t)(c.dc[k] & 0xFF)); f.push_back((uint8_t)((c.dc[k] >> 8) & 0xFF)); }
    }
    put64(f, payload.bits);
    f.insert(f.end(), payload.buf.begin(), payload.buf.end());
    return f;
}

// ---------------------------------------------------------------------------
// Decoder pieces
// ---------------------------------------------------------------------------
struct BitReader
{
    const uint8_t* p; uint64_t pos = 0, limit = 0; bool ok = true;
    int bit()
    {
        if (pos >= limit) { ok = false; return 0; }
        int b = (p[pos >> 3] >> (7 - (pos & 7))) & 1;
        ++pos;
        return b;
    }
    int bits(int n)
    {
        if (n < 0 || n > 31) { ok = false; return 0; }
        int v = 0;
        while (n--) v = (v << 1) | bit();
        return v;
    }
};

struct DecTable
{
    uint64_t first[65]; int idx[65]; int cnt[65]; std::vector<int> sym;
    void build(const HuffTable& t)
    {
        std::vector<int> s;
        for (int i = 0; i < 256; ++i) if (t.len[i]) s.push_back(i);
        std::sort(s.begin(), s.end(), [&](int a, int b)
                  { return t.len[a] != t.len[b] ? t.len[a] < t.len[b] : a < b; });
        sym = s;
        std::memset(cnt, 0, sizeof(cnt)); std::memset(first, 0, sizeof(first)); std::memset(idx, 0, sizeof(idx));
        for (int i = 0; i < (int)s.size(); ++i)
        {
            int L = t.len[s[i]];
            if (cnt[L] == 0) { first[L] = t.code[s[i]]; idx[L] = i; }
            cnt[L]++;
        }
    }
    int decode(BitReader& br) const
    {
        uint64_t code = 0;
        for (int L = 1; L <= 64; ++L)
        {
            code = (code << 1) | br.bit();
            if (cnt[L] && code >= first[L] && code - first[L] < (uint64_t)cnt[L])
                return sym[idx[L] + (int)(code - first[L])];
        }
        return -1;
    }
};

inline uint32_t rd32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
inline uint64_t rd64(const uint8_t* p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

struct Header
{
    int w = 0, h = 0;
    int quality = 0;
    Tables T;
    DecTable D[4];
    std::vector<ChunkInfo> chunks;
    const uint8_t* payload = nullptr;
    uint64_t payloadBits = 0;
};

// D0: read the header and build the Huffman decoding tables (small, serial)
inline bool parse_file(const std::vector<uint8_t>& f, Header& H)
{
    if (f.size() < 13 || std::memcmp(f.data(), "IMGC", 4) != 0) return false;
    H.w = (int)rd32(&f[4]); H.h = (int)rd32(&f[8]);
    H.quality = f[12];
    if (H.w <= 0 || H.h <= 0 || H.quality < 1 || H.quality > 100) return false;
    size_t pos = 13;
    for (int t = 0; t < 4; ++t)
    {
        if (pos + 2 > f.size()) return false;
        H.T.t[t] = HuffTable();
        int n = f[pos] | (f[pos + 1] << 8); pos += 2;
        if (n > 256 || pos + (size_t)n * 2 > f.size()) return false;
        for (int i = 0; i < n; ++i)
        {
            uint8_t symbol = f[pos], length = f[pos + 1];
            if (length == 0 || length > 63 || H.T.t[t].len[symbol]) return false;
            H.T.t[t].len[symbol] = length;
            pos += 2;
        }
        assign_canonical(H.T.t[t]);
        H.D[t].build(H.T.t[t]);
    }
    if (pos + 4 > f.size()) return false;
    uint32_t nch = rd32(&f[pos]); pos += 4;
    if (nch == 0 || pos > f.size() - 8 || (size_t)nch > (f.size() - pos - 8) / 14) return false;
    H.chunks.resize(nch);
    for (uint32_t k = 0; k < nch; ++k)
    {
        H.chunks[k].bitOffset = rd64(&f[pos]); pos += 8;
        for (int c = 0; c < 3; ++c) { H.chunks[k].dc[c] = (int16_t)(f[pos] | (f[pos + 1] << 8)); pos += 2; }
    }
    H.payloadBits = rd64(&f[pos]); pos += 8;
    if (H.payloadBits > (uint64_t)(f.size() - pos) * 8) return false;
    H.payload = f.data() + pos;
    return true;
}

// ===========================================================================
//  D1  -  entropy decoding of ONE chunk of blocks.
//  A chunk starts at a known bit position with a known DC predictor
//  => chunks are independent of each other
// ===========================================================================
inline bool decode_chunk(const Header& H, const Geometry& g, int k, Coefs& C)
{
    BitReader br{H.payload, H.chunks[k].bitOffset, H.payloadBits};
    if (br.pos > br.limit) return false;
    int prev[3] = {H.chunks[k].dc[0], H.chunks[k].dc[1], H.chunks[k].dc[2]};
    int s = k * CHUNK_BLOCKS, e = std::min(g.nb, s + CHUNK_BLOCKS);
    for (int b = s; b < e; ++b)
        for (int c = 0; c < 3; ++c)
        {
            int cls = c > 0 ? 2 : 0;
            int16_t* blk = &C.c[c][(size_t)b * 64];
            std::memset(blk, 0, 64 * sizeof(int16_t));
            int sz = H.D[cls].decode(br);
            if (sz < 0 || sz > 15 || !br.ok) return false;
            int d = 0;
            if (sz > 0) { d = br.bits(sz); if (d < (1 << (sz - 1))) d -= (1 << sz) - 1; }
            blk[0] = (int16_t)(prev[c] + d);
            prev[c] = blk[0];
            int kk = 1;
            while (kk < 64)
            {
                int sy = H.D[cls + 1].decode(br);
                if (sy < 0 || !br.ok) return false;
                if (sy == 0x00) break;
                if (sy == 0xF0) { kk += 16; if (kk > 64) return false; continue; }
                int z = sy & 15;
                if (z == 0) return false;
                kk += sy >> 4;
                if (kk > 63) return false;
                int v = br.bits(z);
                if (!br.ok) return false;
                if (v < (1 << (z - 1))) v -= (1 << z) - 1;
                blk[ZZ[kk]] = (int16_t)v; ++kk;
            }
        }
    return true;
}

// ===========================================================================
//  D2  -  dequantization + inverse DCT of ONE 8x8 block of ONE component
//  Reads only its own 64 coefficients, writes only its own 64 pixels
//  => every block is independent
// ===========================================================================
inline void idct_block(const Coefs& C, const Geometry& g, int c, int b, Planes& P)
{
    const int* q = QT[c > 0 ? 1 : 0];
    const int16_t* blk = &C.c[c][(size_t)b * 64];
    float F[64], tmp[64];
    for (int i = 0; i < 64; ++i) F[i] = (float)blk[i] * (float)q[i];
    for (int y = 0; y < 8; ++y)
        for (int v = 0; v < 8; ++v)
        {
            float s = 0;
            for (int u = 0; u < 8; ++u) s += DCTM[u][y] * F[u * 8 + v];
            tmp[y * 8 + v] = s;
        }
    int bx = b % g.bw, by = b / g.bw;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
        {
            float s = 0;
            for (int v = 0; v < 8; ++v) s += tmp[y * 8 + v] * DCTM[v][x];
            P.p[c][(size_t)(by * 8 + y) * g.pw + bx * 8 + x] = s;
        }
}

// ===========================================================================
//  D3  -  colour conversion YCbCr -> RGB for ONE output row
//  Every pixel depends only on the same pixel of the three planes => independent
// ===========================================================================
inline void color_back_row(const Planes& P, const Geometry& g, int y, Image& out)
{
    auto cl = [](float v) { return (uint8_t)std::clamp((int)lrintf(v), 0, 255); };
    for (int x = 0; x < g.w; ++x)
    {
        size_t o = (size_t)y * g.pw + x;
        float Y = P.p[0][o] + 128.0f, Cb = P.p[1][o], Cr = P.p[2][o];
        uint8_t* d = &out.rgb[((size_t)y * g.w + x) * 3];
        d[0] = cl(Y + 1.402f * Cr);
        d[1] = cl(Y - 0.344136f * Cb - 0.714136f * Cr);
        d[2] = cl(Y + 1.772f * Cb);
    }
}

// ---------------------------------------------------------------------------
// Quality metrics (same definitions as the main project: MSE over all RGB
// values, PSNR from MSE, simple global SSIM on the grayscale image)
// ---------------------------------------------------------------------------
inline double calc_mse(const Image& a, const Image& b)
{
    double s = 0;
    for (size_t i = 0; i < a.rgb.size(); ++i) { double d = (double)a.rgb[i] - b.rgb[i]; s += d * d; }
    return s / (double)a.rgb.size();
}
inline double calc_psnr(double mse) { return mse == 0 ? INFINITY : 10.0 * std::log10(255.0 * 255.0 / mse); }
inline double calc_ssim(const Image& a, const Image& b)
{
    size_t n = (size_t)a.w * a.h;
    if (n == 0 || b.w != a.w || b.h != a.h || b.rgb.size() != a.rgb.size()) return -1;
    if (n == 1) return a.rgb == b.rgb ? 1.0 : 0.0;
    auto gray = [](const uint8_t* p) { return (double)((p[0] * 4899 + p[1] * 9617 + p[2] * 1868 + 8192) >> 14); };
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; ++i) { mx += gray(&a.rgb[i * 3]); my += gray(&b.rgb[i * 3]); }
    mx /= n; my /= n;
    double vx = 0, vy = 0, cv = 0;
    for (size_t i = 0; i < n; ++i)
    {
        double dx = gray(&a.rgb[i * 3]) - mx, dy = gray(&b.rgb[i * 3]) - my;
        vx += dx * dx; vy += dy * dy; cv += dx * dy;
    }
    vx /= (n - 1); vy /= (n - 1); cv /= (n - 1);
    const double C1 = (0.01 * 255) * (0.01 * 255), C2 = (0.03 * 255) * (0.03 * 255);
    return ((2 * mx * my + C1) * (2 * cv + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2));
}

inline uint64_t fnv1a(const std::vector<uint8_t>& d)
{
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t c : d) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// ---------------------------------------------------------------------------
// Shared "after the timed work" part: write .bin / .png, check, print report,
// append one CSV row.
// ---------------------------------------------------------------------------
inline int finish_run(const std::string& mode, int threads, const std::string& imagePath,
                       const Image& im, const Geometry& g, const Coefs& enc,
                       const std::vector<uint8_t>& file,
                       const Timings& mean, double bestTotal,
                       const Coefs& dec, const Image& rec, bool decodeOk,
                       const DTimings& dmean, double dbest,
                       int reps, double tRead)
{
    fs::create_directories("../output/single");
    fs::create_directories("../results");
    std::string tag = (mode == "sequential") ? "sequential" : "parallel_t" + std::to_string(threads);
    std::string outputTag = fs::path(imagePath).stem().string() + "_" + tag;
    std::string binPath = "../output/single/" + outputTag + ".bin";
    std::string pngPath = "../output/single/" + outputTag + "_reconstructed.png";
    { std::ofstream o(binPath, std::ios::binary); o.write((const char*)file.data(), (std::streamsize)file.size()); }

    bool lossless = decodeOk && dec.c[0] == enc.c[0] && dec.c[1] == enc.c[1] && dec.c[2] == enc.c[2];
    double mse = decodeOk ? calc_mse(im, rec) : -1, psnr = calc_psnr(mse), ssim = decodeOk ? calc_ssim(im, rec) : -1;
    if (decodeOk) save_png(pngPath, rec);

    double raw = (double)im.rgb.size();
    uintmax_t origBytes = fs::exists(imagePath) ? fs::file_size(imagePath) : 0;
    uint64_t sum = fnv1a(file);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== " << mode << (mode == "parallel" ? " (" + std::to_string(threads) + " threads)" : "")
              << " | " << imagePath << " | " << g.w << "x" << g.h << " | " << reps << " rep(s) ===\n";
    std::cout << "COMPRESSION                        mean time (ms)\n";
    std::cout << "  1 colour conversion  RGB->YCbCr  " << std::setw(10) << mean.color * 1e3 << "\n"
              << "  2 DCT + quantization             " << std::setw(10) << mean.dct   * 1e3 << "\n"
              << "  3 symbol statistics (histogram)  " << std::setw(10) << mean.hist  * 1e3 << "\n"
              << "  4 build Huffman tables           " << std::setw(10) << mean.huff  * 1e3 << "\n"
              << "  5 Huffman encoding               " << std::setw(10) << mean.enc   * 1e3 << "\n"
              << "  6 merge streams + write header   " << std::setw(10) << mean.merge * 1e3 << "\n"
              << "  TOTAL compression (mean)         " << std::setw(10) << mean.total * 1e3 << "\n"
              << "  TOTAL compression (best rep)     " << std::setw(10) << bestTotal * 1e3 << "\n";
    std::cout << "DECOMPRESSION                      mean time (ms)\n"
              << "  D0 parse header + decode tables  " << std::setw(10) << dmean.parse   * 1e3 << "\n"
              << "  D1 Huffman decoding              " << std::setw(10) << dmean.entropy * 1e3 << "\n"
              << "  D2 dequantization + inverse DCT  " << std::setw(10) << dmean.idct    * 1e3 << "\n"
              << "  D3 colour conversion YCbCr->RGB  " << std::setw(10) << dmean.color   * 1e3 << "\n"
              << "  TOTAL decompression (mean)       " << std::setw(10) << dmean.total   * 1e3 << "\n"
              << "  TOTAL decompression (best rep)   " << std::setw(10) << dbest * 1e3 << "\n"
              << "  (reading the file, not counted)  " << std::setw(10) << tRead * 1e3 << "\n";
    std::cout << "Compressed size : " << file.size() << " bytes   ratio (raw RGB / compressed) : "
              << raw / file.size() << "   space saving : " << 100.0 * (1 - file.size() / raw) << " %\n";
    std::cout << "Original file  : " << origBytes << " bytes   ratio (original file / compressed) : "
              << (double)origBytes / file.size() << "\n";
    std::cout << "Decoded coefficients identical to encoder : " << (lossless ? "YES" : "NO  <-- ERROR") << "\n";
    std::cout << "MSE " << mse << "   PSNR " << psnr << " dB   SSIM " << ssim << "\n";
    std::cout << "Bitstream checksum (FNV-1a) : " << std::hex << sum << std::dec << "\n";
    std::cout << "Saved: " << binPath << " , " << pngPath << "\n";

    bool fresh = !fs::exists("../results/single_results.csv");
    std::ofstream csv("../results/single_results.csv", std::ios::app);
    if (fresh)
        csv << "Mode,Threads,Image,Width,Height,Reps,Read_ms,Color_ms,DCT_Quant_ms,Hist_ms,Huffman_ms,"
               "Encode_ms,Merge_ms,Total_mean_ms,Total_best_ms,Compressed_bytes,Ratio,MSE,PSNR_dB,SSIM,"
               "Checksum,Original_bytes,Parse_ms,Decode_ms,IDCT_ms,ColorBack_ms,Decomp_mean_ms,Decomp_best_ms\n";
    csv << std::fixed << std::setprecision(4) << mode << "," << threads << "," << imagePath << "," << g.w << ","
        << g.h << "," << reps << "," << tRead * 1e3 << "," << mean.color * 1e3 << "," << mean.dct * 1e3 << ","
        << mean.hist * 1e3 << "," << mean.huff * 1e3 << "," << mean.enc * 1e3 << "," << mean.merge * 1e3 << ","
        << mean.total * 1e3 << "," << bestTotal * 1e3 << "," << file.size() << "," << raw / file.size() << ","
        << mse << "," << psnr << "," << ssim << "," << std::hex << sum << std::dec << "," << origBytes << ","
        << dmean.parse * 1e3 << "," << dmean.entropy * 1e3 << "," << dmean.idct * 1e3 << ","
        << dmean.color * 1e3 << "," << dmean.total * 1e3 << "," << dbest * 1e3 << "\n";
    return lossless ? 0 : 1;
}
