// ============================================================================
//  sequential_single.cpp  -  compress AND decompress ONE image, sequentially
//
//  Build :  g++ sequential_single.cpp -o sequential_single -std=c++17 -O2 -ffp-contract=off
//  Run   :  ./sequential_single <image> [repetitions]
//
//  Every stage runs in a single thread.  Each stage is timed so we can see
//  which ones are expensive (candidates for parallelization).
// ============================================================================
#include "codec_common.h"

// ----------------------------------------------------------------------------
//  COMPRESSION
// ----------------------------------------------------------------------------
static std::vector<uint8_t> compress(const Image& im, const Geometry& g, Planes& P, Coefs& C, Timings& T)
{
    double t0 = now_sec();

    // ---- Stage 1: colour conversion -------------------------------------
    for (int y = 0; y < g.ph; ++y) color_row(im, g, P, y);
    double t1 = now_sec();

    // ---- Stage 2: DCT + quantization of every 8x8 block ------------------
    for (int b = 0; b < g.nb; ++b)
        for (int c = 0; c < 3; ++c)
            dct_quant_block(P, g, c, b, &C.c[c][(size_t)b * 64]);
    double t2 = now_sec();

    // ---- Stage 3: count how often every symbol occurs --------------------
    Hist H; HistSink hs{&H};
    for (int b = 0; b < g.nb; ++b)
        for (int c = 0; c < 3; ++c)
            walk_block(hs, c, &C.c[c][(size_t)b * 64], b ? C.c[c][(size_t)(b - 1) * 64] : 0);
    double t3 = now_sec();

    // ---- Stage 4: build the Huffman tables --------------------------------
    Tables Tab;
    for (int t = 0; t < 4; ++t) build_huffman(H.n[t], Tab.t[t]);
    double t4 = now_sec();

    // ---- Stage 5: Huffman-encode all blocks into ONE bit stream -----------
    //      (the stream is split into chunks of CHUNK_BLOCKS blocks; we only
    //       note where each chunk starts so the decoder can use it)
    int nchunks = (g.nb + CHUNK_BLOCKS - 1) / CHUNK_BLOCKS;
    std::vector<ChunkInfo> chunks(nchunks);
    BitWriter bw; EncSink es{&bw, &Tab};
    for (int k = 0; k < nchunks; ++k)
    {
        int s = k * CHUNK_BLOCKS, e = std::min(g.nb, s + CHUNK_BLOCKS);
        chunks[k].bitOffset = bw.bits;
        for (int c = 0; c < 3; ++c) chunks[k].dc[c] = s ? C.c[c][(size_t)(s - 1) * 64] : 0;
        for (int b = s; b < e; ++b)
            for (int c = 0; c < 3; ++c)
                walk_block(es, c, &C.c[c][(size_t)b * 64], b ? C.c[c][(size_t)(b - 1) * 64] : 0);
    }
    bw.finish();
    double t5 = now_sec();

    // ---- Stage 6: write header + payload ----------------------------------
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

    // ---- D0: read header, build Huffman decoding tables --------------------
    if (!parse_file(file, H)) return false;
    int expectedChunks = (g.nb + CHUNK_BLOCKS - 1) / CHUNK_BLOCKS;
    if (H.w != g.w || H.h != g.h || H.quality != QUALITY ||
        (int)H.chunks.size() != expectedChunks) return false;
    double t1 = now_sec();

    // ---- D1: Huffman decoding, chunk after chunk ---------------------------
    bool ok = true;
    for (int k = 0; k < (int)H.chunks.size(); ++k) ok &= decode_chunk(H, g, k, D);
    double t2 = now_sec();

    // ---- D2: dequantization + inverse DCT of every block --------------------
    for (int b = 0; b < g.nb; ++b)
        for (int c = 0; c < 3; ++c)
            idct_block(D, g, c, b, Q);
    double t3 = now_sec();

    // ---- D3: colour conversion YCbCr -> RGB ----------------------------------
    for (int y = 0; y < g.h; ++y) color_back_row(Q, g, y, rec);
    double t4 = now_sec();

    T.parse = t1 - t0; T.entropy = t2 - t1; T.idct = t3 - t2; T.color = t4 - t3; T.total = t4 - t0;
    return ok;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::cout << "Usage: sequential_single <image> [repetitions]\n"; return 1; }
    std::string path = argv[1];
    int reps = argc > 2 ? std::max(1, atoi(argv[2])) : 1;

    init_codec(QUALITY);

    double tr = now_sec();
    Image im;
    if (!load_image(path, im)) { std::cerr << "Cannot read image: " << path << "\n"; return 1; }
    tr = now_sec() - tr;

    Geometry g(im.w, im.h);
    Planes P(g); Coefs C(g);                       // allocated once, outside the timed stages
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

    return finish_run("sequential", 1, path, im, g, C, file, csum, cbest, D, rec, ok, dsum, dbest, reps, tr);
}
