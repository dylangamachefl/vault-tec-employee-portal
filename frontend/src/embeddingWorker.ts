/**
 * Embedding Web Worker — runs all heavy model work off the main UI thread.
 *
 * Messages IN:  { type: 'embed', query: string }
 * Messages OUT: { type: 'ready' }              — model loaded, first time
 *               { type: 'result', vector: number[] }
 *               { type: 'error',  message: string }
 */

import { pipeline, env, type Tensor } from '@xenova/transformers';

// Serve WASM/model files from the CDN instead of bundling them.
env.allowLocalModels = false;

type EmbeddingPipeline = Awaited<ReturnType<typeof pipeline>>;
let embedder: EmbeddingPipeline | null = null;

async function getEmbedder(): Promise<EmbeddingPipeline> {
    if (embedder) return embedder;
    embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
    );
    return embedder;
}

// Signal ready after model is warmed up on first load
getEmbedder()
    .then(() => self.postMessage({ type: 'ready' }))
    .catch((err: Error) =>
        self.postMessage({ type: 'error', message: err.message }),
    );

self.onmessage = async (event: MessageEvent<{ type: string; query: string; nonce?: string }>) => {
    if (event.data.type !== 'embed') return;

    try {
        const model = await getEmbedder();
        const output = await model(event.data.query, {
            pooling: 'mean',
            // @ts-expect-error: Xenova type definition for normalize is overly restrictive
            normalize: true,
        }) as Tensor;

        // output.data is a Float32Array — convert to plain number[] for postMessage
        const vector = Array.from(output.data as Float32Array);
        self.postMessage({ type: 'result', vector, nonce: event.data.nonce });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown embedding error';
        self.postMessage({ type: 'error', message, nonce: event.data.nonce });
    }
};
