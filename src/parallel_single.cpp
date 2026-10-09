// ============================================================================
//  parallel_single.cpp  -  compress AND decompress ONE image, parallelised
//                          INSIDE the image
//
//  Build :  g++ parallel_single.cpp -o parallel_single -std=c++17 -O2 -fopenmp -ffp-contract=off
//  Run   :  ./parallel_single <image> <threads> [repetitions]
//
//  Same pipeline and same helper functions as sequential_single.cpp; only the
//  stage loops are different.
//
//  COMPRESSION - which stages can run in parallel, and why:
//   1 colour conversion : every pixel is computed from the same input pixel only
//                         -> rows are independent                          PARALLEL
//   2 DCT + quantization: a block reads its own 64 pixels and writes its own 64
//                         coefficients -> blocks are independent           PARALLEL
//   3 symbol statistics : counting is independent per block; each thread counts
//                         into a private histogram, then they are added up PARALLEL
//   4 build Huffman     : one small tree over <=256 symbols, needs ALL counts  SERIAL
//   5 Huffman encoding  : the code of a block does not depend on the bits before
//                         it, so chunks of blocks are encoded into private
//                         bit buffers                                      PARALLEL
//   6 merge             : chunk buffers are joined in order (the position of a
//                         chunk depends on the sizes of all earlier ones)    SERIAL
//
//  DECOMPRESSION
//   D0 parse header     : reads the file header, builds decode tables        SERIAL
//   D1 Huffman decoding : the header stores where each chunk starts and the DC
//                         value before it, so chunks are decoded independently PARALLEL
//   D2 dequant + IDCT   : a block only touches its own 64 values            PARALLEL
//   D3 colour conversion: each pixel depends only on the same pixel         PARALLEL
// ============================================================================
#include <omp.h>
#include "codec_common.h"

// ----------------------------------------------------------------------------
//  COMPRESSION
// ----------------------------------------------------------------------------
static std::vector<uint8_t> compress(const Image& im, const Geometry& g, Planes& P, Coefs& C, Timings& T)
{
    double t0 = now_sec();

    // ---- Stage 1: colour conversion (parallel over image rows) -------------
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < g.ph; ++y) color_row(im, g, P, y);
    double t1 = now_sec();

    // ---- Stage 2: DCT + quantization (parallel over 8x8 blocks) ------------
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < g.nb; ++b)
        for (int c = 0; c < 3; ++c)
            dct_quant_block(P, g, c, b, &C.c[c][(size_t)b * 64]);
    double t2 = now_sec();

    // ---- Stage 3: symbol statistics (private histograms, then reduce) ------
    Hist H;
    #pragma omp parallel
    {
        Hist local; HistSink hs{&local};
        #pragma omp for schedule(static) nowait
        for (int b = 0; b < g.nb; ++b)
            for (int c = 0; c < 3; ++c)
                walk_block(hs, c, &C.c[c][(size_t)b * 64], b ? C.c[c][(size_t)(b - 1) * 64] : 0);
        #pragma omp critical
        H.add(local);                               // integer addition: order does not matter
    }
    double t3 = now_sec();

    // ---- Stage 4: build the Huffman tables (serial) ---------------------------
    Tables Tab;
    for (int t = 0; t < 4; ++t) build_huffman(H.n[t], Tab.t[t]);
    double t4 = now_sec();

    // ---- Stage 5: encode independent chunks of blocks in parallel -------------
    int nchunks = (g.nb + CHUNK_BLOCKS - 1) / CHUNK_BLOCKS;
    std::vector<BitWriter> parts(nchunks);
    std::vector<ChunkInfo> chunks(nchunks);
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < nchunks; ++k)
    {
        int s = k * CHUNK_BLOCKS, e = std::min(g.nb, s + CHUNK_BLOCKS);
        for (int c = 0; c < 3; ++c) chunks[k].dc[c] = s ? C.c[c][(size_t)(s - 1) * 64] : 0;
        EncSink es{&parts[k], &Tab};
        for (int b = s; b < e; ++b)
            for (int c = 0; c < 3; ++c)
                walk_block(es, c, &C.c[c][(size_t)b * 64], b ? C.c[c][(size_t)(b - 1) * 64] : 0);
        parts[k].finish();
    }
    double t5 = now_sec();

    // ---- Stage 6: join the chunks in order + write header (serial) -----------
    BitWriter bw;
    for (int k = 0; k < nchunks; ++k) { chunks[k].bitOffset = bw.bits; bw.append(parts[k]); }
    bw.finish();
    std::vector<uint8_t> file = make_file(g, Tab, bw, chunks);
    double t6 = now_sec();

    T.color = t1 - t0; T.dct = t2 - t1; T.hist = t3 - t2; T.huff = t4 - t3;
    T.enc = t5 - t4;   T.merge = t6 - t5; T.total = t6 - t0;
    return file;
}

// ----------------------------------------------------------------------------
//  DECOMPRESSION
// ----------------------------------------------------------------------------
static bool decompress(const std::vector<uint8_t>& file, const Geometry& g, Header& H,
                       Coefs& D, Planes& Q, Image& rec, DTimings& T)
{
    double t0 = now_sec();

    // ---- D0: read header, build Huffman decoding tables (serial) ------------
    if (!parse_file(file, H)) return false;
    int expectedChunks = (g.nb + CHUNK_BLOCKS - 1) / CHUNK_BLOCKS;
    if (H.w != g.w || H.h != g.h || H.quality != QUALITY ||
        (int)H.chunks.size() != expectedChunks) return false;
    double t1 = now_sec();

    // ---- D1: Huffman decoding, one independent chunk per loop iteration -----
    int ok = 1;
    int nchunks = (int)H.chunks.size();
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < nchunks; ++k)
        if (!decode_chunk(H, g, k, D))
        {
            #pragma omp atomic write
            ok = 0;
        }
    double t2 = now_sec();

    // ---- D2: dequantization + inverse DCT (parallel over blocks) -------------
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < g.nb; ++b)
        for (int c = 0; c < 3; ++c)
            idct_block(D, g, c, b, Q);
    double t3 = now_sec();

    // ---- D3: colour conversion YCbCr -> RGB (parallel over rows) -------------
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < g.h; ++y) color_back_row(Q, g, y, rec);
    double t4 = now_sec();

    T.parse = t1 - t0; T.entropy = t2 - t1; T.idct = t3 - t2; T.color = t4 - t3; T.total = t4 - t0;
    return ok != 0;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::cout << "Usage: parallel_single <image> <threads> [repetitions]\n"; return 1; }
    std::string path = argv[1];
    int threads = std::max(1, atoi(argv[2]));
    int reps = argc > 3 ? std::max(1, atoi(argv[3])) : 1;
    omp_set_num_threads(threads);

    init_codec(QUALITY);

    double tr = now_sec();
    Image im;
    if (!load_image(path, im)) { std::cerr << "Cannot read image: " << path << "\n"; return 1; }
    tr = now_sec() - tr;

    Geometry g(im.w, im.h);
    Planes P(g); Coefs C(g);
    Planes Q(g); Coefs D(g); Header H;
    Image rec; rec.w = g.w; rec.h = g.h; rec.rgb.assign((size_t)g.w * g.h * 3, 0);

    Timings csum, cone; double cbest = 1e300; std::vector<uint8_t> file;
    for (int r = 0; r < reps; ++r)
    {
        file = compress(im, g, P, C, cone);
        csum.add(cone); cbest = std::min(cbest, cone.total);
    }
    csum.scale(1.0 / reps);

    DTimings dsum, done; double dbest = 1e300; bool ok = true;
    for (int r = 0; r < reps; ++r)
    {
        ok &= decompress(file, g, H, D, Q, rec, done);
        dsum.add(done); dbest = std::min(dbest, done.total);
    }
    dsum.scale(1.0 / reps);

    return finish_run("parallel", threads, path, im, g, C, file, csum, cbest, D, rec, ok, dsum, dbest, reps, tr);
}
