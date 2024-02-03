use itertools::Itertools;
use rand::seq::SliceRandom;
use tinyvec::TinyVec;

// https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers/blob/3e09136cc1bde6bba73ca5813a3f6c982cd5555a/dataloader.py#L73-L83
pub fn batch<'a>(
    data: &[[TinyVec<[i64; 128]>; 2]],
    tokens_per_batch: usize,
) -> Vec<Vec<&[TinyVec<[i64; 128]>; 2]>> {
    let groups = data.into_iter().group_by(|[_, tgt]| tgt.len());
    let chunks = groups
        .into_iter()
        .map(|(_, g)| g.collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let mut all_batches = vec![];
    for mut chunk in chunks {
        chunk.sort_by_key(|[src, _]| src.len());
        let seqs_per_batch = tokens_per_batch / chunk[0][1].len();
        for i in (0..chunk.len()).step_by(seqs_per_batch) {
            all_batches.push(chunk[i..(i + seqs_per_batch).min(chunk.len())].to_vec());
        }
    }
    all_batches.shuffle(&mut rand::thread_rng());
    all_batches
}
